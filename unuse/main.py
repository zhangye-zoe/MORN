#!/usr/bin/env python
# coding: utf-8 -*-

import os
import json
import time
import glob
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import dgl

from models.morn import MORN
from utils import (
    parse_args_with_config,
    resolve_paths,
    ensure_nid,
    ensure_edge_weight,
    load_survival_from_csv,
    count_params,
    NLLSurvLoss,
    train_one_fold,
    dump_all_attn_to_files,
)

from utils.pathway_contrast import PathwaySupConLoss


# ======================================================================================
# ✅ ABLATION SWITCHES (改这里就行，不需要改 argparse)
# ======================================================================================
# ABLATION = {
#     # --- modalities ---
#     "use_wsi": True,

#     "use_cnv": True,
#     "use_methy": True,
#     "use_mrna": True,
#     "use_mirna": True,

#     # --- regulation edges ---
#     "use_mti": True,          # miRNA -> mRNA
#     "use_pathway_hub": True,  # CNV/Methy -> hub mRNA

#     # --- safety / debug ---
#     "print_selected_etypes": True,
# }


ABLATION = {
    # --- modalities ---
    "use_wsi": False,

    "use_cnv": False,
    "use_methy": False,
    "use_mrna": True,
    "use_mirna": False,

    # --- regulation edges ---
    "use_mti": True,          # miRNA -> mRNA
    "use_pathway_hub": True,  # CNV/Methy -> hub mRNA

    # --- safety / debug ---
    "print_selected_etypes": True,
}

# 你要做的典型消融（举例）：
# (1) 只用 mRNA：use_mrna=True, 其他组学&reg&WSI=False
# (2) 加其他三组学：use_cnv/use_methy/use_mirna=True, reg可先关后开
# (3) 再加WSI：use_wsi=True
# (4) 单独一个组学：只开那个组学即可


# ======================================================================================
# Helpers: load edge_groups / build subgraph while preserving all nodes
# ======================================================================================

def _try_load_edge_groups(args, meta: dict) -> Optional[dict]:
    """
    Locate and load edge_groups.pt.
    Priority:
      1) meta["edge_groups_pt"]
      2) <args.data_dir>/edge_groups.pt
      3) <args.data_dir>/../edge_groups.pt
    """
    candidates = []
    if isinstance(meta, dict) and meta.get("edge_groups_pt", None):
        candidates.append(meta["edge_groups_pt"])

    candidates.append(os.path.join(args.data_dir, "edge_groups.pt"))
    candidates.append(os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "edge_groups.pt"))

    for p in candidates:
        if p and os.path.isfile(p):
            try:
                eg = torch.load(p, map_location="cpu")
                print(f"[ABL] Loaded edge_groups: {p}")
                return eg
            except Exception as e:
                print(f"[ABL][WARN] Failed to load {p}: {e}")

    print("[ABL][WARN] edge_groups.pt not found -> cannot do clean ablation by etype groups.")
    return None


def _unique_keep_order(seq: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _select_etypes_from_edge_groups(edge_groups: dict, sw: dict) -> List[Tuple[str, str, str]]:
    """
    edge_groups structure:
      {
        "wsi": [ (patient,has_wsi,wsi), (wsi,belongs_to,patient) ],
        "patient_omics": { "CNV":[...], "Methy":[...], "mRNA":[...], "miRNA":[...] },
        "regulation": { "mti":[...], "pathway_hub":[...] },
        "all": [...],
        "canonical_etypes": [...]
      }
    """
    selected = []

    # WSI
    if sw.get("use_wsi", True):
        selected += edge_groups.get("wsi", [])

    # patient <-> omics
    po = edge_groups.get("patient_omics", {})
    if sw.get("use_cnv", True):
        selected += po.get("CNV", [])
    if sw.get("use_methy", True):
        selected += po.get("Methy", [])
    if sw.get("use_mrna", True):
        selected += po.get("mRNA", [])
    if sw.get("use_mirna", True):
        selected += po.get("miRNA", [])

    # regulation
    reg = edge_groups.get("regulation", {})
    if sw.get("use_mti", True):
        selected += reg.get("mti", [])
    if sw.get("use_pathway_hub", True):
        selected += reg.get("pathway_hub", [])

    return _unique_keep_order(selected)


def build_preserve_nodes_subgraph(
    G: dgl.DGLHeteroGraph,
    selected_etypes: List[Tuple[str, str, str]],
) -> dgl.DGLHeteroGraph:
    """
    关键：保证所有 ntype 的节点数与原图一致（不丢 patient/wsi），否则 split/label 会错位。
    我们手工构造 heterograph(data_dict, num_nodes_dict=原图) 并拷贝 node/edge features。
    """
    if len(selected_etypes) == 0:
        raise ValueError("[ABL] selected_etypes is empty. At least keep some edges (e.g. patient<->mRNA or patient<->wsi).")

    num_nodes_dict = {nt: G.num_nodes(nt) for nt in G.ntypes}

    data_dict = {}
    for et in selected_etypes:
        if et not in G.canonical_etypes:
            continue
        u, v = G.edges(etype=et)
        data_dict[et] = (u, v)

    if len(data_dict) == 0:
        raise ValueError("[ABL] None of selected_etypes exist in G.canonical_etypes.")

    G2 = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    # copy node data
    for ntype in G.ntypes:
        for k, val in G.nodes[ntype].data.items():
            # keep dtype/device; G2 initially on cpu
            G2.nodes[ntype].data[k] = val

    # copy edge data (only for kept etypes)
    for et in G2.canonical_etypes:
        for k, val in G.edges[et].data.items():
            G2.edges[et].data[k] = val

    return G2


# ======================================================================================
# CL helper (保持你原来的)
# ======================================================================================

def _try_load_pathway_maps(args, meta: dict):
    """
    Locate and load pathway_maps.pt.
    Priority:
      1) meta["pathway_maps_pt"] (if you saved it in preprocessing)
      2) <args.data_dir>/pathway_maps.pt
      3) <args.data_dir>/../pathway_maps.pt
    """
    candidates = []
    if isinstance(meta, dict) and meta.get("pathway_maps_pt", None):
        candidates.append(meta["pathway_maps_pt"])

    candidates.append(os.path.join(args.data_dir, "pathway_maps.pt"))
    candidates.append(os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "pathway_maps.pt"))

    for p in candidates:
        if p and os.path.isfile(p):
            try:
                pm = torch.load(p, map_location="cpu")
                print(f"[CL] Loaded pathway_maps: {p}")
                return pm
            except Exception as e:
                print(f"[CL][WARN] Failed to load {p}: {e}")

    print("[CL][WARN] pathway_maps.pt not found -> CL disabled.")
    return None


def main():
    args = parse_args_with_config()

    # 如果你未来把开关加进 argparse，可以在这里自动覆盖：
    #（不加也没关系，默认用 ABLATION dict）
    sw = dict(ABLATION)
    for k in list(sw.keys()):
        if hasattr(args, k):
            sw[k] = bool(getattr(args, k))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ds, graph_path, label_csv, out_dir = resolve_paths(args)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[DATASET] {ds}")
    if getattr(args, "config", None):
        print(f"[CONFIG] {args.config}")
    print(f"[PATH] data_dir   = {args.data_dir}")
    print(f"[PATH] graph_path = {graph_path}")
    print(f"[PATH] label_csv  = {label_csv}")
    print(f"[PATH] out_dir    = {out_dir}")

    assert os.path.isfile(graph_path), f"Not found graph: {graph_path}"
    assert os.path.isfile(label_csv), f"Not found label: {label_csv}"

    graphs, _ = dgl.load_graphs(graph_path)
    G_full = graphs[0]
    target = args.target_ntype

    # y_disc
    assert "label" in G_full.nodes[target].data, f"Need G.nodes['{target}'].data['label']"
    y_disc_cpu = G_full.nodes[target].data["label"].long()
    labeled = (y_disc_cpu >= 0)
    if not labeled.any():
        raise RuntimeError("No labeled patients (label==-1 for all).")
    n_bins = int(y_disc_cpu[labeled].max().item() + 1)
    print(f"[INFO] n_bins={n_bins}")

    # WSI patch dim (only if wsi is in full graph)
    if "wsi" in G_full.ntypes:
        assert "wsi_patches" in G_full.nodes["wsi"].data, "Need G.nodes['wsi'].data['wsi_patches'] (N,K,D)"
        wsi_patches = G_full.nodes["wsi"].data["wsi_patches"]
        assert wsi_patches.dim() == 3, f"wsi_patches must be 3D (N,K,D), got {tuple(wsi_patches.shape)}"
        wsi_feat_dim = int(wsi_patches.shape[2])   # D
        wsi_topk_patch = int(wsi_patches.shape[1]) # K
        print(f"[INFO] wsi_feat_dim={wsi_feat_dim}, wsi_topk_patch={wsi_topk_patch}")
        if "wsi_patch_mask" not in G_full.nodes["wsi"].data:
            print("[WARN] missing G.nodes['wsi'].data['wsi_patch_mask']; will treat all K patches as valid.")
    else:
        wsi_feat_dim = 0
        wsi_topk_patch = 0

    ensure_nid(G_full)
    ensure_edge_weight(G_full, args.edge_weight_key)

    fold_dirs = sorted(glob.glob(os.path.join(args.data_dir, args.fold_glob)))
    fold_dirs = [d for d in fold_dirs if os.path.isdir(d)]
    if len(fold_dirs) == 0:
        raise FileNotFoundError(f"No fold dirs under {args.data_dir} with glob={args.fold_glob}")

    # -------- CL args (safe defaults) --------
    lambda_cl = float(getattr(args, "lambda_cl", 0.1))
    temperature = float(getattr(args, "cl_temperature", 0.1))
    pathways_per_step = int(getattr(args, "cl_pathways_per_step", 128))
    min_genes_per_pathway = int(getattr(args, "cl_min_genes_per_pathway", 3))
    proj_dim = int(getattr(args, "cl_proj_dim", 128))

    per_fold = []
    test_scores_for_stats = []

    for fold_idx, fold_dir in enumerate(fold_dirs):
        fold_name = os.path.basename(fold_dir)

        split_candidates = sorted(glob.glob(os.path.join(fold_dir, f"*{args.split_pt_suffix}")))
        meta_candidates  = sorted(glob.glob(os.path.join(fold_dir, f"*{args.meta_json_suffix}")))
        if len(split_candidates) == 0:
            raise FileNotFoundError(f"[{fold_name}] no split pt (*{args.split_pt_suffix})")
        if len(meta_candidates) == 0:
            raise FileNotFoundError(f"[{fold_name}] no meta json (*{args.meta_json_suffix})")

        split_path = split_candidates[0]
        meta_path  = meta_candidates[0]

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        patients = meta.get("patients")
        if patients is None:
            raise KeyError(f"[{fold_name}] meta json missing 'patients'")

        # =========================
        # ✅ Build ablation subgraph
        # =========================
        edge_groups = _try_load_edge_groups(args, meta)
        if edge_groups is None:
            # fallback: no edge_groups -> use full graph but still respect use_wsi in model
            print("[ABL] edge_groups not found -> use FULL graph (no etype pruning).")
            G_use = G_full
            selected_etypes = list(G_full.canonical_etypes)
        else:
            selected_etypes = _select_etypes_from_edge_groups(edge_groups, sw)
            if sw.get("print_selected_etypes", True):
                print(f"\n[ABL] switches = {sw}")
                print(f"[ABL] selected canonical etypes ({len(selected_etypes)}):")
                for et in selected_etypes:
                    print(f"  - {et}")

            G_use = build_preserve_nodes_subgraph(G_full, selected_etypes)

        # survival aligned
        event_time_cpu, censorship_cpu = load_survival_from_csv(label_csv, patients)

        # split idx
        split = torch.load(split_path, map_location="cpu")
        train_idx_cpu = split["train_idx"].long()
        val_idx_cpu   = split["val_idx"].long()
        test_idx_cpu  = split["test_idx"].long()

        def _filter_labeled(idx: torch.Tensor) -> torch.Tensor:
            return idx[y_disc_cpu[idx] >= 0]

        train_idx_cpu = _filter_labeled(train_idx_cpu)
        val_idx_cpu   = _filter_labeled(val_idx_cpu)
        test_idx_cpu  = _filter_labeled(test_idx_cpu)

        # move to device
        G_dev = G_use.to(device)
        y_disc = y_disc_cpu.to(device)
        event_time = event_time_cpu.to(device)
        censorship = censorship_cpu.to(device)
        train_idx = train_idx_cpu.to(device)
        val_idx   = val_idx_cpu.to(device)
        test_idx  = test_idx_cpu.to(device)

        node_dict = {ntype: i for i, ntype in enumerate(G_dev.ntypes)}
        edge_dict = {c_etype: i for i, c_etype in enumerate(G_dev.canonical_etypes)}

        # model layers
        n_layers_omics = getattr(args, "n_layers_omics", None)
        n_layers_cross = getattr(args, "n_layers_cross", None)
        if n_layers_omics is None:
            n_layers_omics = getattr(args, "n_layers", 2)
        if n_layers_cross is None:
            n_layers_cross = 1

        # ✅ 如果图里没有 wsi 或者开关关掉，模型内部会自动 use_wsi=False
        model = MORN(
            G=G_dev,
            node_dict=node_dict,
            edge_dict=edge_dict,
            n_hid=args.n_hid,
            n_out=n_bins,
            n_layers_omics=n_layers_omics,
            n_layers_cross=n_layers_cross,
            n_heads=args.n_heads,
            use_norm=True,
            wsi_patch_dim=(wsi_feat_dim if "wsi" in G_dev.ntypes else 0),
            edge_weight_key=args.edge_weight_key,
            edge_weight_mode=args.edge_weight_mode,
            dropout=getattr(args, "dropout", 0.2),
            use_wsi=bool(sw.get("use_wsi", True)),
        ).to(device)

        # =========================
        # CL module
        # =========================
        pathway_maps = _try_load_pathway_maps(args, meta)

        cl_module = None
        if pathway_maps is not None and lambda_cl > 0:
            # 如果消融把 gene_* 都关了（或图里没有），CL 没意义，直接禁用
            has_any_gene = any(nt.startswith("gene_") for nt in G_dev.ntypes)
            if not has_any_gene:
                print("[CL] graph has no gene_* ntypes -> CL disabled.")
            else:
                omics_ntypes = pathway_maps.get("omics_ntypes", ["gene_CNV", "gene_Methy", "gene_mRNA"])
                # 过滤掉不在图里的 omics types
                omics_ntypes = [x for x in omics_ntypes if x in G_dev.ntypes]
                if len(omics_ntypes) == 0:
                    print("[CL] no omics_ntypes exist in current graph -> CL disabled.")
                else:
                    cl_module = PathwaySupConLoss(
                        hid_dim=model.n_hid,
                        omics_ntypes=omics_ntypes,
                        proj_dim=proj_dim,
                        temperature=temperature,
                        min_genes_per_pathway=min_genes_per_pathway,
                        pathways_per_step=pathways_per_step,
                        dropout=0.1,
                    ).to(device)
                    print(
                        f"[CL] enabled | lambda={lambda_cl} | T={temperature} | pathways_per_step={pathways_per_step} | "
                        f"min_genes={min_genes_per_pathway} | proj_dim={proj_dim} | omics={omics_ntypes}"
                    )
        else:
            print("[CL] disabled (no pathway_maps or lambda_cl<=0)")

        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)

        # optimizer includes both model + cl_module params (critical)
        params = list(model.parameters())
        if cl_module is not None:
            params += list(cl_module.parameters())
        optimizer = torch.optim.AdamW(params, lr=args.max_lr, weight_decay=args.weight_decay)

        # epoch-level OneCycle (since you step scheduler once per epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=args.n_epoch,
            max_lr=args.max_lr
        )

        print("\n" + "=" * 110)
        print(
            f"[{ds} | FOLD {fold_name}] ({fold_idx+1}/{len(fold_dirs)}) "
            f"train/val/test={len(train_idx_cpu)}/{len(val_idx_cpu)}/{len(test_idx_cpu)} | "
            f"params={count_params(model)}"
        )
        print(f"[GRAPH] ntypes={G_dev.ntypes}")
        print(f"[GRAPH] etypes={len(G_dev.canonical_etypes)}")
        print("=" * 110)

        t0 = time.time()
        fold_metrics = train_one_fold(
            model=model,
            G=G_dev,
            target_ntype=target,
            y_disc=y_disc,
            event_time=event_time,
            censorship=censorship,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epoch=args.n_epoch,
            eval_every=args.eval_every,
            clip=args.clip,
            # CL
            pathway_maps=pathway_maps,
            cl_module=cl_module,
            lambda_cl=lambda_cl,
        )

        fold_metrics.update({
            "dataset": ds,
            "fold": fold_name,
            "split_path": split_path,
            "meta_path": meta_path,
            "train_n": int(len(train_idx_cpu)),
            "val_n": int(len(val_idx_cpu)),
            "test_n": int(len(test_idx_cpu)),
            "elapsed_sec": float(time.time() - t0),
            "test_for_cv": float(fold_metrics["best_test_tracked"]),
            "ablation_switches": sw,
            "selected_etypes": [list(x) for x in G_use.canonical_etypes],
        })

        per_fold.append({k: v for k, v in fold_metrics.items() if k != "best_state"})
        test_scores_for_stats.append(fold_metrics["best_test_tracked"])

        # dump attention (保持你原来的功能)
        if args.dump_attn == 1:
            dump_root = os.path.join(out_dir, args.dump_dirname, ds, fold_name)
            os.makedirs(dump_root, exist_ok=True)

            if args.dump_attn_at == "best" and fold_metrics.get("best_state") is not None:
                # best_state might include "_cl_module" -- strip it for model loading
                state = {k: v for k, v in fold_metrics["best_state"].items() if k != "_cl_module"}
                model.load_state_dict(state, strict=True)

            model.eval()
            with torch.no_grad():
                out = model(G_dev, target, return_attn=True, return_patch_attn=True)
                if isinstance(out, (tuple, list)) and len(out) >= 3:
                    _logits, all_attn, patch_attn = out[0], out[1], out[2]
                elif isinstance(out, (tuple, list)) and len(out) >= 2:
                    _logits, all_attn = out[0], out[1]
                    patch_attn = None
                else:
                    raise RuntimeError("model(return_attn=True) must return (logits, all_attn, ...)")

            dump_all_attn_to_files(
                G_cpu=G_dev.to("cpu"),
                all_attn=all_attn,
                out_dir=dump_root,
                meta={
                    "dataset": ds,
                    "fold": fold_name,
                    "dump_at": args.dump_attn_at,
                    "best_epoch": fold_metrics.get("best_epoch", -1),
                    "best_val": fold_metrics.get("best_val"),
                    "best_test_tracked": fold_metrics.get("best_test_tracked"),
                    "final_test": fold_metrics.get("final_test"),
                    "edge_weight_key": args.edge_weight_key,
                    "edge_weight_mode": args.edge_weight_mode,
                    "n_hid": args.n_hid,
                    "n_layers_omics": int(n_layers_omics),
                    "n_layers_cross": int(n_layers_cross),
                    "n_heads": args.n_heads,
                    "seed": args.seed,
                    "config": getattr(args, "config", None),
                    "wsi_feat_dim": int(wsi_feat_dim),
                    "wsi_topk_patch": int(wsi_topk_patch),
                    "lambda_cl": float(lambda_cl),
                    "cl_temperature": float(temperature),
                    "cl_pathways_per_step": int(pathways_per_step),
                    "cl_min_genes_per_pathway": int(min_genes_per_pathway),
                    "cl_proj_dim": int(proj_dim),
                    "ablation_switches": sw,
                    "canonical_etypes_used": [list(x) for x in G_dev.canonical_etypes],
                    "has_patch_attn": bool(patch_attn is not None),
                },
                edge_weight_key=args.edge_weight_key,
                save_edge_weight=bool(args.save_edge_w_in_dump),
            )

            # 额外：如果你想把 patch_attn 也保存下来（可视化会很方便）
            if patch_attn is not None:
                torch.save(patch_attn.detach().cpu(), os.path.join(dump_root, "patch_attn.pt"))

        # save fold results
        fold_out = os.path.join(out_dir, ds, fold_name)
        os.makedirs(fold_out, exist_ok=True)
        fold_json = {k: v for k, v in fold_metrics.items() if k != "best_state"}
        with open(os.path.join(fold_out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(fold_json, f, indent=2, ensure_ascii=False)

        if args.save_model:
            torch.save(model.state_dict(), os.path.join(fold_out, "model_final.pt"))
            if cl_module is not None:
                torch.save(cl_module.state_dict(), os.path.join(fold_out, "cl_module_final.pt"))

        del model, cl_module, optimizer, scheduler, loss_fn
        torch.cuda.empty_cache()

    scores = np.array(test_scores_for_stats, dtype=np.float64)
    mean = float(np.nanmean(scores))
    std = float(np.nanstd(scores, ddof=1)) if np.isfinite(scores).sum() > 1 else float("nan")

    summary = {
        "dataset": ds,
        "metric": "c-index",
        "scores": [float(x) for x in scores.tolist()],
        "mean": mean,
        "std": std,
        "args": vars(args),
        "folds": per_fold,
        "ablation_switches": sw,
    }

    summary_path = os.path.join(out_dir, ds, "cv_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[CV SUMMARY] {ds} Mean±Std = {mean:.4f} ± {std:.4f}")
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
