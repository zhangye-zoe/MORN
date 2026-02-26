#!/usr/bin/env python
# coding: utf-8
"""
Train MORN_V2 (omics warmup + optional WSI token mixer) with:
- patient has NO embedding (not random ID)
- forbid patient as message SOURCE (no patient->gene leakage)
- biological type-constrained random walk + supervised contrastive loss
- stage-wise training: warmup omics then enable WSI mixer

Now supports two tasks via args.task:
  - survival (default): c-index + NLLSurvLoss
  - grading: macro-F1 + CrossEntropyLoss
"""

import os
import json
import time
import glob
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import dgl

from models.morn import MORN
from utils.train_one_fold import train_one_fold

from utils import (
    parse_args_with_config,
    resolve_paths,
    ensure_nid,
    ensure_edge_weight,
    load_survival_from_csv,
    count_params,
    NLLSurvLoss,
    dump_all_attn_to_files,
)

# ======================================================================================
# ✅ ABLATION SWITCHES (改这里就行，不需要改 argparse)
# ======================================================================================
ABLATION = {
    # --- modalities ---
    "use_wsi": True,

    "use_cnv": True,
    "use_methy": True,
    "use_mrna": True,
    "use_mirna": True,

    # --- regulation edges ---
    "use_mti": True,          # miRNA -> mRNA
    "use_pathway_hub": True,  # CNV/Methy -> hub mRNA

    # --- safety / debug ---
    "print_selected_etypes": True,
}

# ======================================================================================
# Helpers: load edge_groups / build subgraph while preserving all nodes
# ======================================================================================

def _try_load_edge_groups(args, meta: dict):
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

                # ✅ NEW: coerce to expected structure
                eg2 = _coerce_edge_groups_to_expected(eg)

                # （可选）打印一下确认类型
                po = eg2.get("patient_omics", None)
                print(f"[ABL] edge_groups.patient_omics type = {type(po)} "
                      f"keys={list(po.keys()) if isinstance(po, dict) else None}")

                return eg2
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

from typing import Dict, List, Tuple, Any

CanonEtype = Tuple[str, str, str]

def _coerce_edge_groups_to_expected(eg: Any) -> Dict[str, Any]:
    """
    Convert edge_groups to the structure expected by _select_etypes_from_edge_groups():

    {
      "wsi": [canonical_etype...],
      "patient_omics": {"CNV": [...], "Methy": [...], "mRNA": [...], "miRNA": [...]},
      "regulation": {"mti": [...], "pathway_hub": [...]}
    }

    Works even if:
      - eg is a list of canonical etypes
      - eg["patient_omics"] is a list (old format)
      - missing keys
    """
    # case: eg is directly a list of etypes
    if isinstance(eg, list):
        eg = {"all": eg}

    if not isinstance(eg, dict):
        raise TypeError(f"edge_groups must be dict/list, got {type(eg)}")

    out = {
        "wsi": [],
        "patient_omics": {"CNV": [], "Methy": [], "mRNA": [], "miRNA": []},
        "regulation": {"mti": [], "pathway_hub": []},
    }

    # ---------- helper: bucket patient<->omics ----------
    def _bucket_patient_omics(et: CanonEtype):
        s, e, d = et
        if s != "patient" and d != "patient":
            return False
        other = d if s == "patient" else s

        # ✅ 按你项目真实 ntype 命名来改（如果不同）
        if other == "gene_CNV":
            out["patient_omics"]["CNV"].append(et); return True
        if other == "gene_Methy":
            out["patient_omics"]["Methy"].append(et); return True
        if other == "gene_mRNA":
            out["patient_omics"]["mRNA"].append(et); return True
        if other == "gene_miRNA":
            out["patient_omics"]["miRNA"].append(et); return True
        return False

    # ---------- helper: bucket regulation ----------
    def _bucket_reg(et: CanonEtype):
        s, e, d = et
        # mti: miRNA <-> mRNA（按你项目实际方向/etype可再细化）
        if (s == "gene_miRNA" and d == "gene_mRNA") or (s == "gene_mRNA" and d == "gene_miRNA"):
            out["regulation"]["mti"].append(et); return True
        # pathway hub: pathway <-> *
        if s == "pathway" or d == "pathway":
            out["regulation"]["pathway_hub"].append(et); return True
        return False

    # ---------- 1) 如果已经是新格式，直接尽量拷贝 ----------
    # wsi
    if isinstance(eg.get("wsi", None), list):
        out["wsi"] = list(eg.get("wsi", []))

    # patient_omics：可能是 dict（新）也可能是 list（旧）
    po = eg.get("patient_omics", None)
    if isinstance(po, dict):
        for k in ["CNV", "Methy", "mRNA", "miRNA"]:
            if isinstance(po.get(k, None), list):
                out["patient_omics"][k] = list(po.get(k, []))
    elif isinstance(po, list):
        # 旧格式：把 list 里的 etype 重新分桶
        for et in po:
            _bucket_patient_omics(tuple(et))

    # regulation：可能是 dict（新）也可能是 list（旧）
    reg = eg.get("regulation", None)
    if isinstance(reg, dict):
        for k in ["mti", "pathway_hub"]:
            if isinstance(reg.get(k, None), list):
                out["regulation"][k] = list(reg.get(k, []))
    elif isinstance(reg, list):
        # 旧格式：全部当 mti（或你也可以都重新分桶）
        for et in reg:
            et = tuple(et)
            if not _bucket_reg(et):
                out["regulation"]["mti"].append(et)

    # ---------- 2) 兜底：如果 eg 还有 “all / other / omics_omics …” 等旧 key，把它们扫一遍补齐 ----------
    for key, val in eg.items():
        if not isinstance(val, list):
            continue
        for et in val:
            et = tuple(et)
            s, e, d = et
            if s == "wsi" or d == "wsi":
                if et not in out["wsi"]:
                    out["wsi"].append(et)
                continue
            if _bucket_patient_omics(et):
                continue
            if _bucket_reg(et):
                continue

    return out


def _select_etypes_from_edge_groups(edge_groups: dict, sw: dict) -> List[Tuple[str, str, str]]:
    selected = []

    if sw.get("use_wsi", True):
        selected += edge_groups.get("wsi", [])

    po = edge_groups.get("patient_omics", {})
    if sw.get("use_cnv", True):
        selected += po.get("CNV", [])
    if sw.get("use_methy", True):
        selected += po.get("Methy", [])
    if sw.get("use_mrna", True):
        selected += po.get("mRNA", [])
    if sw.get("use_mirna", True):
        selected += po.get("miRNA", [])

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

    for ntype in G.ntypes:
        for k, val in G.nodes[ntype].data.items():
            G2.nodes[ntype].data[k] = val

    for et in G2.canonical_etypes:
        for k, val in G.edges[et].data.items():
            G2.edges[et].data[k] = val

    return G2


# ======================================================================================
# Biological type constraints for walks
# ======================================================================================

def get_default_allowed_next_types(G: dgl.DGLHeteroGraph) -> Dict[str, List[str]]:
    nt = set(G.ntypes)
    allowed = {}

    def add(src, dsts):
        if src in nt:
            ok = [d for d in dsts if d in nt]
            if len(ok) > 0:
                allowed[src] = ok

    add("patient", ["gene_mRNA"])
    add("gene_mRNA", ["gene_miRNA", "gene_Methy", "gene_CNV"])
    add("gene_miRNA", ["gene_mRNA"])
    add("gene_Methy", ["gene_mRNA"])

    if len(allowed) == 0:
        print("[CL][WARN] default allowed_next_types matched none of your ntypes. Fallback to graph-neighbor transitions.")
        return {}

    return allowed


def get_default_metapaths(G: dgl.DGLHeteroGraph) -> Optional[List[List[str]]]:
    nt = set(G.ntypes)
    cand = [
        ["patient", "gene_mRNA", "gene_miRNA"],
        ["patient", "gene_mRNA", "gene_Methy"],
        ["patient", "gene_mRNA", "gene_CNV"],
    ]
    kept = [m for m in cand if all(x in nt for x in m)]
    return kept if len(kept) > 0 else None


# ======================================================================================
# Main
# ======================================================================================

def main():
    args = parse_args_with_config()

    # -----------------------
    # task switch (NEW)
    # -----------------------
    task = str(getattr(args, "task", "survival")).lower()  # survival | grading
    assert task in ("survival", "grading"), f"--task must be survival|grading, got {task}"

    # grading options (NEW)
    grade_label_key = str(getattr(args, "grade_label_key", "label"))  # node data key
    num_grades_arg = int(getattr(args, "num_grades", -1))             # if -1 -> infer from labels
    use_class_weight = int(getattr(args, "use_class_weight", 0))      # 0/1
    label_smoothing = float(getattr(args, "label_smoothing", 0.0))    # float

    # print('grade label key', grade_label_key)

    # allow overriding ABLATION by argparse (if exists)
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
    print(f"[TASK] {task}")
    print(f"[PATH] data_dir   = {args.data_dir}")
    print(f"[PATH] graph_path = {graph_path}")
    print(f"[PATH] label_csv  = {label_csv}")
    print(f"[PATH] out_dir    = {out_dir}")

    assert os.path.isfile(graph_path), f"Not found graph: {graph_path}"
    if task == "survival":
        assert os.path.isfile(label_csv), f"Not found label: {label_csv}"

    graphs, _ = dgl.load_graphs(graph_path)
    G_full = graphs[0]
    target = args.target_ntype

    # -----------------------
    # labels by task (NEW)
    # -----------------------
    if task == "survival":
        assert "label" in G_full.nodes[target].data, f"Need G.nodes['{target}'].data['label']"
        y_disc_cpu = G_full.nodes[target].data["label"].long()
        labeled = (y_disc_cpu >= 0)
        if not labeled.any():
            raise RuntimeError("No labeled patients (label==-1 for all).")
        n_out = int(y_disc_cpu[labeled].max().item() + 1)
        print(f"[INFO] n_bins={n_out}")
    else:
        assert grade_label_key in G_full.nodes[target].data, (
            f"Need G.nodes['{target}'].data['{grade_label_key}'] for grading"
        )
        y_grade_cpu = G_full.nodes[target].data[grade_label_key].long()
        labeled = (y_grade_cpu >= 0)
        if not labeled.any():
            raise RuntimeError(f"No labeled samples for grading ({grade_label_key}==-1 for all).")
        if num_grades_arg > 0:
            n_out = num_grades_arg
        else:
            n_out = int(y_grade_cpu[labeled].max().item() + 1)
        print(f"[INFO] num_classes={n_out} (label_key={grade_label_key})")

    # WSI patch dim (only if graph has wsi)
    if "wsi" in G_full.ntypes:
        assert "wsi_patches" in G_full.nodes["wsi"].data, "Need G.nodes['wsi'].data['wsi_patches'] (N,K,D)"
        wsi_patches = G_full.nodes["wsi"].data["wsi_patches"]
        assert wsi_patches.dim() == 3, f"wsi_patches must be 3D (N,K,D), got {tuple(wsi_patches.shape)}"
        wsi_feat_dim = int(wsi_patches.shape[2])
        wsi_topk_patch = int(wsi_patches.shape[1])
        print(f"[INFO] wsi_feat_dim={wsi_feat_dim}, wsi_topk_patch={wsi_topk_patch}")
        if "wsi_patch_mask" not in G_full.nodes["wsi"].data:
            print("[WARN] missing wsi_patch_mask; will treat all patches as valid.")
    else:
        wsi_feat_dim = 0
        wsi_topk_patch = 0

    ensure_nid(G_full)
    ensure_edge_weight(G_full, args.edge_weight_key)

    # folds
    fold_dirs = sorted(glob.glob(os.path.join(args.data_dir, args.fold_glob)))
    fold_dirs = [d for d in fold_dirs if os.path.isdir(d)]
    if len(fold_dirs) == 0:
        raise FileNotFoundError(f"No fold dirs under {args.data_dir} with glob={args.fold_glob}")

    # # ---- v2 hyperparams ----
    warmup_epochs = int(getattr(args, "warmup_epochs", 10))
    lambda_walk_cl = float(getattr(args, "lambda_walk_cl", 0.1))
    cl_temperature = float(getattr(args, "cl_temperature", 0.1))
    walk_len = int(getattr(args, "walk_len", 4))
    walks_per_patient = int(getattr(args, "walks_per_patient", 4))
    cl_batch_size = int(getattr(args, "cl_batch_size", 0))


    # ---- v2 hyperparams ----
    # warmup_epochs = int(getattr(args, "warmup_epochs", 10))
    # lambda_walk_cl = float(getattr(args, "lambda_walk_cl", 1.0))
    # cl_temperature = float(getattr(args, "cl_temperature", 0.5))
    # walk_len = int(getattr(args, "walk_len", 3))
    # walks_per_patient = int(getattr(args, "walks_per_patient", 3))
    # cl_batch_size = int(getattr(args, "cl_batch_size", 8))

    per_fold = []
    test_scores_for_stats = []
    test_scores_final = []

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
        # Build ablation subgraph
        # =========================
        edge_groups = _try_load_edge_groups(args, meta)
        if edge_groups is None:
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

        # survival aligned (only for survival)
        if task == "survival":
            event_time_cpu, censorship_cpu = load_survival_from_csv(label_csv, patients)
        else:
            event_time_cpu, censorship_cpu = None, None

        # split
        split = torch.load(split_path, map_location="cpu")
        train_idx_cpu = split["train_idx"].long()
        val_idx_cpu   = split["val_idx"].long()
        test_idx_cpu  = split["test_idx"].long()

        def _filter_labeled(idx: torch.Tensor) -> torch.Tensor:
            if task == "survival":
                return idx[y_disc_cpu[idx] >= 0]
            else:
                return idx[y_grade_cpu[idx] >= 0]

        train_idx_cpu = _filter_labeled(train_idx_cpu)
        val_idx_cpu   = _filter_labeled(val_idx_cpu)
        test_idx_cpu  = _filter_labeled(test_idx_cpu)

        # to device
        G_dev = G_use.to(device)
        train_idx = train_idx_cpu.to(device)
        val_idx   = val_idx_cpu.to(device)
        test_idx  = test_idx_cpu.to(device)

        if task == "survival":
            y_disc = y_disc_cpu.to(device)
            event_time = event_time_cpu.to(device)
            censorship = censorship_cpu.to(device)
            y_grade = None
        else:
            y_grade = y_grade_cpu.to(device)
            y_disc = None
            event_time = None
            censorship = None

        node_dict = {ntype: i for i, ntype in enumerate(G_dev.ntypes)}
        edge_dict = {c_etype: i for i, c_etype in enumerate(G_dev.canonical_etypes)}

        # layers
        n_layers_omics = getattr(args, "n_layers_omics", None)
        if n_layers_omics is None:
            n_layers_omics = getattr(args, "n_layers", 2)
        n_layers_omics = int(n_layers_omics)

        # choose omics tokens (gene_* types)
        omics_token_ntypes = [nt for nt in G_dev.ntypes if nt.startswith("gene_")]
        omics_token_ntypes = sorted(omics_token_ntypes)

        # model
        model = MORN(
            G=G_dev,
            node_dict=node_dict,
            edge_dict=edge_dict,
            n_hid=args.n_hid,
            n_out=n_out,
            n_layers_omics=n_layers_omics,
            n_layers_cross=0,
            n_heads=args.n_heads,
            use_norm=True,
            wsi_patch_dim=(wsi_feat_dim if "wsi" in G_dev.ntypes else 0),
            dropout=float(getattr(args, "dropout", 0.2)),
            edge_weight_key=args.edge_weight_key,
            edge_weight_mode=args.edge_weight_mode,
            use_wsi=bool(sw.get("use_wsi", True)),
            omics_token_ntypes=omics_token_ntypes,
        ).to(device)

        # -----------------------
        # loss_fn by task (NEW)
        # -----------------------
        if task == "survival":
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        else:
            ce_weight = None
            if use_class_weight == 1:
                y_tr = y_grade_cpu[train_idx_cpu]
                y_tr = y_tr[y_tr >= 0]
                counts = torch.bincount(y_tr, minlength=n_out).float()
                ce_weight = (counts.sum() / (counts + 1e-6))
                ce_weight = ce_weight / ce_weight.mean()
                ce_weight = ce_weight.to(device)
                print("[CE] counts:", counts.tolist())
                print("[CE] weight:", ce_weight.detach().cpu().tolist())
            print('grading')

            loss_fn = torch.nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=label_smoothing)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.max_lr,
            weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=args.n_epoch,
            max_lr=args.max_lr
        )

        # biological constraints for walk
        metapaths = get_default_metapaths(G_dev)
        allowed_next_types = get_default_allowed_next_types(G_dev)
        if metapaths is not None:
            print(f"[CL] Using metapaths (count={len(metapaths)}): {metapaths}")
        else:
            print(f"[CL] Using allowed_next_types: {allowed_next_types if len(allowed_next_types)>0 else 'fallback(auto)'}")

        print("\n" + "=" * 110)
        print(
            f"[{ds} | FOLD {fold_name}] ({fold_idx+1}/{len(fold_dirs)}) "
            f"train/val/test={len(train_idx_cpu)}/{len(val_idx_cpu)}/{len(test_idx_cpu)} | "
            f"params={count_params(model)}"
        )
        print(f"[GRAPH] ntypes={G_dev.ntypes}")
        print(f"[GRAPH] etypes={len(G_dev.canonical_etypes)}")
        print(
            f"[V2] warmup_epochs={warmup_epochs} | lambda_walk_cl={lambda_walk_cl} | "
            f"T={cl_temperature} | walk_len={walk_len} | walks_per_patient={walks_per_patient} | cl_batch={cl_batch_size}"
        )
        print("=" * 110)

        t0 = time.time()
        fold_metrics = train_one_fold(
            model=model,
            G=G_dev,
            target_ntype=target,

            # NEW
            task=task,

            # survival
            y_disc=y_disc,
            event_time=event_time,
            censorship=censorship,

            # grading
            y_grade=y_grade,
            num_classes=(n_out if task == "grading" else None),

            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epoch=args.n_epoch,
            eval_every=args.eval_every,
            clip=args.clip,

            warmup_epochs=warmup_epochs,

            lambda_walk_cl=lambda_walk_cl,
            cl_temperature=cl_temperature,
            walk_len=walk_len,
            walks_per_patient=walks_per_patient,
            batch_size=cl_batch_size,

            allowed_next_types=(allowed_next_types if metapaths is None else None),
            metapath_types_list=metapaths,
            edge_weight_key=args.edge_weight_key,
        )

        fold_metrics.update({
            "dataset": ds,
            "fold": fold_name,
            "task": task,
            "split_path": split_path,
            "meta_path": meta_path,
            "train_n": int(len(train_idx_cpu)),
            "val_n": int(len(val_idx_cpu)),
            "test_n": int(len(test_idx_cpu)),
            "elapsed_sec": float(time.time() - t0),
            "test_for_cv": float(fold_metrics["best_test_tracked"]),
            "test_final": float(fold_metrics["final_test"]),
            "ablation_switches": sw,
            "canonical_etypes_used": [list(x) for x in G_dev.canonical_etypes],
            "warmup_epochs": warmup_epochs,
            "lambda_walk_cl": lambda_walk_cl,
            "cl_temperature": cl_temperature,
            "walk_len": walk_len,
            "walks_per_patient": walks_per_patient,
            "cl_batch_size": cl_batch_size,
            "omics_token_ntypes": omics_token_ntypes,
            "n_out": int(n_out),
            "grade_label_key": (grade_label_key if task == "grading" else None),
            "use_class_weight": use_class_weight,
            "label_smoothing": label_smoothing,
        })

        per_fold.append({k: v for k, v in fold_metrics.items() if k != "best_state"})
        test_scores_for_stats.append(fold_metrics["best_test_tracked"])
        test_scores_final.append(fold_metrics["final_test"])

        # dump attention (保留原逻辑)
        if args.dump_attn == 1:
            dump_root = os.path.join(out_dir, args.dump_dirname, ds, fold_name)
            os.makedirs(dump_root, exist_ok=True)

            if args.dump_attn_at == "best" and fold_metrics.get("best_state") is not None:
                model.load_state_dict(fold_metrics["best_state"], strict=True)

            model.eval()
            with torch.no_grad():
                out = model(G_dev, target, return_attn=True, return_patch_attn=True)
                logits, all_attn, patch_attn = out[0], out[1], out[2] if len(out) >= 3 else None

            dump_all_attn_to_files(
                G_cpu=G_dev.to("cpu"),
                all_attn=all_attn,
                out_dir=dump_root,
                meta={
                    "dataset": ds,
                    "fold": fold_name,
                    "task": task,
                    "dump_at": args.dump_attn_at,
                    "best_epoch": fold_metrics.get("best_epoch", -1),
                    "best_val": fold_metrics.get("best_val"),
                    "best_test_tracked": fold_metrics.get("best_test_tracked"),
                    "final_test": fold_metrics.get("final_test"),
                    "edge_weight_key": args.edge_weight_key,
                    "edge_weight_mode": args.edge_weight_mode,
                    "n_hid": args.n_hid,
                    "n_layers_omics": int(n_layers_omics),
                    "n_heads": args.n_heads,
                    "seed": args.seed,
                    "config": getattr(args, "config", None),
                    "wsi_feat_dim": int(wsi_feat_dim),
                    "wsi_topk_patch": int(wsi_topk_patch),
                    "ablation_switches": sw,
                    "warmup_epochs": warmup_epochs,
                    "lambda_walk_cl": lambda_walk_cl,
                    "cl_temperature": cl_temperature,
                    "walk_len": walk_len,
                    "walks_per_patient": walks_per_patient,
                    "omics_token_ntypes": omics_token_ntypes,
                    "has_patch_attn": bool(patch_attn is not None),
                    "n_out": int(n_out),
                    "grade_label_key": (grade_label_key if task == "grading" else None),
                },
                edge_weight_key=args.edge_weight_key,
                save_edge_weight=bool(args.save_edge_w_in_dump),
            )

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

        del model, optimizer, scheduler, loss_fn
        torch.cuda.empty_cache()

    # CV summary
    scores = np.array(test_scores_for_stats, dtype=np.float64)
    mean = float(np.nanmean(scores))
    std = float(np.nanstd(scores, ddof=1)) if np.isfinite(scores).sum() > 1 else float("nan")

    final_scores = np.array(test_scores_final, dtype=np.float64)
    final_mean = float(np.nanmean(final_scores))
    final_std = float(np.nanstd(final_scores, ddof=1)) if np.isfinite(final_scores).sum() > 1 else float("nan")

    metric_name = "c-index" if task == "survival" else "macro-F1"

    summary = {
        "dataset": ds,
        "task": task,
        "metric": metric_name,
        "scores": [float(x) for x in scores.tolist()],
        "mean": mean,
        "std": std,
        "final_mean": final_mean,
        "final_std": final_std,
        "args": vars(args),
        "folds": per_fold,
        "ablation_switches": sw,
        "warmup_epochs": warmup_epochs,
        "lambda_walk_cl": lambda_walk_cl,
        "cl_temperature": cl_temperature,
        "walk_len": walk_len,
        "walks_per_patient": walks_per_patient,
        "cl_batch_size": cl_batch_size,
        "n_out": int(n_out),
        "grade_label_key": (grade_label_key if task == "grading" else None),
        "use_class_weight": use_class_weight,
        "label_smoothing": label_smoothing,
    }

    summary_path = os.path.join(out_dir, ds, "cv_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[CV SUMMARY] {ds} ({metric_name}) Mean±Std = {mean:.4f} ± {std:.4f}")
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
