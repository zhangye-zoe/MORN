#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import dgl

# =============================================================================
# Config (按需改这里)
# =============================================================================

DATASET = "ACC"

ATTN_DIR = "/data5/zhangye/morn/data/processed/ACC_hgt_dataset/cv_results/attn_dump/ACC/splits_0"
GRAPH_BIN = "/data5/zhangye/morn/data/processed/ACC_hgt_dataset/ACC_graph.bin"

# 如果图里没有存 gene symbol，这里用于 fallback 还原 gene 名（建议填 ACC 的 raw 文件）
OMICS_DIR = f"/data5/zhangye/morn/data/raw/{DATASET}"
OMICS_FILES_FALLBACK = {
    "gene_CNV":   os.path.join(OMICS_DIR, f"{DATASET}_CNV_top.csv"),
    "gene_Methy": os.path.join(OMICS_DIR, f"{DATASET}_Methy_top.csv"),
    "gene_mRNA":  os.path.join(OMICS_DIR, f"{DATASET}_mRNA_top.csv"),
    "gene_miRNA": os.path.join(OMICS_DIR, f"{DATASET}_miRNA_top.csv"),
}

# 标签文件（用于选 A/B + patient id 映射 fallback）
LABEL_CSV = f"/data5/zhangye/morn/data/label/{DATASET}_survival_labels.csv"
LABEL_COL = "label_disc"
PATIENT_ID_COL = "sample"

PATIENT_A_ID = "TCGA-OR-A5K5"
PATIENT_B_ID = "TCGA-OR-A5KO"

# WSI 原图根目录（用于裁剪 patch 原图）
# 你的目录通常是 /data5/zhangye/morn/data/wsi/ACC/TCGA-XX-XXXX/*.svs
WSI_ROOT = f"/data5/zhangye/morn/data/wsi/{DATASET}"
# Trident features 根目录（用于找 coords）
WSI_H5_ROOT = f"/data5/zhangye/trident/trident_processed_{DATASET}/20x_256px_0px_overlap/features_uni_v1"

# 输出目录（会保存 png + csv + patch png）
OUT_DIR = "./attn_compare_out_v2"

# 关系定义
MRNA2PAT = ("gene_mRNA", "in_patient", "patient")
UPSTREAM_RELS = [
    ("gene_CNV",   "cnv_to_mrna",   "gene_mRNA"),
    ("gene_Methy", "methy_to_mrna", "gene_mRNA"),
    ("gene_miRNA", "targets",       "gene_mRNA"),
]

# TopK
TOPK_SHOW_PLOT = 5      # 图里画 top5
TOPK_SAVE_CSV  = 20     # CSV 保存 top20

# gate：如果 top gate 太严导致空，会自动 fallback 到 ungated
GATE_TOP_MRNA = 200   # None=不用 gate；建议 100~500

# patch（如果存在 patch-level attention）
TOPK_PATCH = 5
PATCH_SIZE = 256     # trident 20x 256px
PATCH_LEVEL = 0      # openslide level，通常 0 是最高分辨率

# =============================================================================
# Utils
# =============================================================================

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def guess_num_layers(attn_dir: str) -> int:
    layers = set()
    for fn in glob.glob(os.path.join(attn_dir, "attn_layer*__*.pt")):
        m = re.search(r"attn_layer(\d+)__", os.path.basename(fn))
        if m:
            layers.add(int(m.group(1)))
    return (max(layers) + 1) if layers else 0

def find_attn_file(attn_dir: str, layer: int, src: str, rel: str, dst: str) -> Optional[str]:
    exact = os.path.join(attn_dir, f"attn_layer{layer}__{src}__{rel}__{dst}.pt")
    if os.path.exists(exact):
        return exact
    cand = glob.glob(os.path.join(attn_dir, f"attn_layer{layer}__{src}__*{rel}*__{dst}.pt"))
    return cand[0] if len(cand) else None

def load_graph_bin(path: str):
    graphs, _ = dgl.load_graphs(path)
    if len(graphs) == 0:
        raise RuntimeError(f"No graphs in {path}")
    return graphs[0]

@dataclass
class AttnPack:
    kind: str  # "edge" or "dense"
    attn: torch.Tensor
    extra: Dict[str, Any]

def load_attn_pt(path: str) -> AttnPack:
    obj = torch.load(path, map_location="cpu")
    extra: Dict[str, Any] = {}

    if isinstance(obj, torch.Tensor):
        return AttnPack(kind="edge", attn=obj, extra=extra)

    if isinstance(obj, dict):
        keys = set(obj.keys())
        if ("attn_mean" in keys) or ("attn_head" in keys):
            # 用 attn_mean 更稳定（E）
            if "attn_mean" in obj and isinstance(obj["attn_mean"], torch.Tensor):
                t = obj["attn_mean"]
            else:
                t = obj["attn_head"]
            for k in ["canonical_etype", "layer", "src", "dst", "edge_w", "attn_mean", "attn_head"]:
                if k in obj:
                    extra[k] = obj[k]
            return AttnPack(kind="edge", attn=t, extra=extra)

        for k in ["attn", "alpha", "a", "weights"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                extra = {kk: vv for kk, vv in obj.items() if kk != k}
                kind = "dense" if (t.dim() == 2 and t.shape[0] > 10 and t.shape[1] > 10) else "edge"
                return AttnPack(kind=kind, attn=t, extra=extra)

        raise ValueError(f"Unrecognized dict format in {path}. keys={list(obj.keys())}")

    raise ValueError(f"Unrecognized pt type: {type(obj)} in {path}")

def reduce_heads(attn: torch.Tensor) -> torch.Tensor:
    if attn.dim() == 1:
        return attn
    if attn.dim() == 2:
        # [H,E] or [E,H]
        if attn.shape[0] < attn.shape[1]:
            return attn.mean(dim=0)
        else:
            return attn.mean(dim=1)
    if attn.dim() >= 3:
        return attn.mean(dim=0)
    return attn

def plot_compare_bar(title: str,
                     items_a: List[Tuple[str, float]],
                     items_b: List[Tuple[str, float]],
                     out_path: str):
    names = sorted(set([n for n, _ in items_a] + [n for n, _ in items_b]))
    da = {n: s for n, s in items_a}
    db = {n: s for n, s in items_b}
    ya = [da.get(n, 0.0) for n in names]
    yb = [db.get(n, 0.0) for n in names]

    x = np.arange(len(names))
    width = 0.38

    plt.figure(figsize=(max(10, 0.9 * len(names)), 4.8))
    plt.bar(x - width / 2, ya, width, label="Patient A")
    plt.bar(x + width / 2, yb, width, label="Patient B")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def save_path_csv(out_csv: str,
                  layer: int,
                  path_name: str,
                  pid_a: str, la: Any,
                  pid_b: str, lb: Any,
                  top_a: List[Tuple[str, float]],
                  top_b: List[Tuple[str, float]]):
    # 对齐并集
    names = sorted(set([n for n, _ in top_a] + [n for n, _ in top_b]))
    da = {n: s for n, s in top_a}
    db = {n: s for n, s in top_b}
    df = pd.DataFrame({
        "layer": layer,
        "path": path_name,
        "gene": names,
        f"{pid_a}|y={la}": [da.get(n, 0.0) for n in names],
        f"{pid_b}|y={lb}": [db.get(n, 0.0) for n in names],
    })
    df = df.sort_values(by=[f"{pid_a}|y={la}", f"{pid_b}|y={lb}"], ascending=False)
    df.to_csv(out_csv, index=False)

# =============================================================================
# Name mapping (graph first, then fallback from omics csv header)
# =============================================================================

def _try_string_list_from_node_data(x: Any, n: int) -> Optional[List[str]]:
    try:
        if isinstance(x, (list, tuple)) and len(x) == n:
            return [str(v) for v in x]
        if isinstance(x, np.ndarray) and x.shape[0] == n:
            return [str(v) for v in x.tolist()]
        if torch.is_tensor(x) and x.numel() == n:
            # DGL 常见不存字符串 -> 这里多半是整数，没用
            return None
    except Exception:
        return None
    return None

def build_name_map_from_graph(G) -> Dict[str, Dict[int, str]]:
    """
    尝试从 G.nodes[ntype].data 里找真实名字（symbol/name/gene...）
    如果没有，则留空，后续用 fallback。
    """
    name_map: Dict[str, Dict[int, str]] = {}

    gene_keys = ["symbol", "gene", "gene_name", "hugo_symbol", "name"]
    patient_keys = ["patient_id", "tcga_id", "sample", "name", "pid_str"]

    for ntype in G.ntypes:
        n = G.num_nodes(ntype)
        data = G.nodes[ntype].data
        keys_to_try = []

        if ntype == "patient":
            keys_to_try = patient_keys
        elif "gene" in ntype.lower() or "mrna" in ntype.lower() or "mirna" in ntype.lower() or "cnv" in ntype.lower() or "methy" in ntype.lower():
            keys_to_try = gene_keys
        else:
            keys_to_try = ["name", "id"]

        chosen = None
        names = None
        for k in keys_to_try:
            if k in data:
                names = _try_string_list_from_node_data(data[k], n)
                if names is not None:
                    chosen = k
                    break

        if names is None:
            continue

        mp = {i: names[i] for i in range(n)}
        name_map[ntype] = mp
        print(f"[INFO] name_map[{ntype}] from graph node.data['{chosen}']")

    return name_map

def build_name_map_fallback_from_omics_csv(omics_files: Dict[str, str]) -> Dict[str, Dict[int, str]]:
    """
    你的构图代码中 gene 节点顺序就是 cols 顺序（build_patient_gene_edges 返回 cols）
    对 ACC 来说，用 *_top.csv 的列名还原 node id -> gene name 是一致的。
    """
    mp: Dict[str, Dict[int, str]] = {}
    for ntype, fpath in omics_files.items():
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, index_col=0)
        # 原始文件是 gene x patient，列是 patient；行 index 是 gene
        # 但你在构图中用 df.T -> cols 是 gene（来自 df 的 index）
        cols = [str(x) for x in df.index.tolist()]
        mp[ntype] = {i: cols[i] for i in range(len(cols))}
        print(f"[INFO] fallback name_map[{ntype}] from {os.path.basename(fpath)} gene index")
    return mp

def name_items(ntype: str,
               top: List[Tuple[int, float]],
               name_map_graph: Dict[str, Dict[int, str]],
               name_map_fallback: Dict[str, Dict[int, str]]) -> List[Tuple[str, float]]:
    mp = name_map_graph.get(ntype, {})
    fb = name_map_fallback.get(ntype, {})
    out = []
    for nid, sc in top:
        nid = int(nid)
        nm = mp.get(nid, fb.get(nid, f"{ntype}:{nid}"))
        out.append((nm, float(sc)))
    return out

# =============================================================================
# Patient id -> nid mapping
# =============================================================================

def _normalize_pid_like_tcga(x: str) -> str:
    s = str(x).strip()
    if s.startswith("TCGA-"):
        parts = s.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
    if "." in s:
        parts = s.split(".")
        if len(parts) >= 3:
            return "-".join(parts[:3])
    return s

def build_patient_id_to_nid(G, label_csv: str, pid_col: str) -> Dict[str, int]:
    # 优先：patient node data 里如果真的存了字符串
    if "patient" in G.ntypes:
        data = G.nodes["patient"].data
        for key in ["patient_id", "tcga_id", "sample", "name", "pid_str"]:
            if key in data:
                names = _try_string_list_from_node_data(data[key], G.num_nodes("patient"))
                if names is not None and any("TCGA" in s for s in names):
                    return {str(names[i]): i for i in range(len(names))}

    # fallback：用 label_csv 行号（你之前的图一般 patients 列表也是按 wsi_dir 排序，这里不保证一致）
    df = pd.read_csv(label_csv)
    pid2nid = {}
    for i, pid in enumerate(df[pid_col].astype(str).map(_normalize_pid_like_tcga).tolist()):
        pid2nid[pid] = i
    print("[WARN] patient_id->nid uses LABEL_CSV row index fallback. Verify patient ordering!")
    return pid2nid

def get_patient_pair(label_csv: str, pid_col: str, label_col: str, a: str, b: str):
    df = pd.read_csv(label_csv)
    df[pid_col] = df[pid_col].astype(str).map(_normalize_pid_like_tcga)
    la = df.loc[df[pid_col] == a, label_col].values
    lb = df.loc[df[pid_col] == b, label_col].values
    if len(la) != 1 or len(lb) != 1:
        raise ValueError("PATIENT_A_ID / PATIENT_B_ID not found in label csv")
    if la[0] == lb[0]:
        raise ValueError("A and B must have different labels")
    return a, b, la[0], lb[0]

# =============================================================================
# Hierarchical scoring (patient-conditioned)
# =============================================================================

def top_src_for_patient_from_src2patient(attn_pack: AttnPack, patient_nid: int, topk: int):
    extra = attn_pack.extra
    assert "src" in extra and "dst" in extra, "need src/dst in attn pt"

    src = extra["src"]
    dst = extra["dst"]
    if torch.is_tensor(src): src = src.detach().cpu().numpy()
    if torch.is_tensor(dst): dst = dst.detach().cpu().numpy()
    src = np.asarray(src).astype(int)
    dst = np.asarray(dst).astype(int)

    w = reduce_heads(attn_pack.attn).detach().cpu().numpy().astype(float)  # [E]

    keep = np.where(dst == int(patient_nid))[0]
    if len(keep) == 0:
        return [], {}

    agg = {}
    for i in keep:
        s = int(src[i])
        agg[s] = agg.get(s, 0.0) + float(w[i])

    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:topk]
    return top, agg

def propagate_upstream_scores(attn_pack: AttnPack,
                             mrna_score: Dict[int, float],
                             topk: int,
                             gate_top_mrna: Optional[set],
                             debug_tag: str = ""):
    """
    score_X(g) = sum_{g->m} attn(g->m) * s_mRNA(m)
    If gated empty -> fallback ungated automatically.
    """
    extra = attn_pack.extra
    assert "src" in extra and "dst" in extra, "need src/dst in attn pt"

    src = extra["src"]
    dst = extra["dst"]
    if torch.is_tensor(src): src = src.detach().cpu().numpy()
    if torch.is_tensor(dst): dst = dst.detach().cpu().numpy()
    src = np.asarray(src).astype(int)
    dst = np.asarray(dst).astype(int)

    w = reduce_heads(attn_pack.attn).detach().cpu().numpy().astype(float)

    mrna_keys = set(mrna_score.keys())
    dst_set = set(dst.tolist())
    overlap = len(dst_set & mrna_keys)

    if debug_tag:
        if gate_top_mrna is not None:
            print(f"[DBG] {debug_tag}: edges={len(dst)} unique_dst={len(dst_set)} mrna_keys={len(mrna_keys)} overlap={overlap} gate={len(gate_top_mrna)} gate_overlap={len(dst_set & set(gate_top_mrna))}")
        else:
            print(f"[DBG] {debug_tag}: edges={len(dst)} unique_dst={len(dst_set)} mrna_keys={len(mrna_keys)} overlap={overlap} gate=None")

    def _run(gate: Optional[set]):
        agg = {}
        for s, m, a in zip(src, dst, w):
            m = int(m)
            if gate is not None and m not in gate:
                continue
            sm = mrna_score.get(m, 0.0)
            if sm <= 0:
                continue
            g = int(s)
            agg[g] = agg.get(g, 0.0) + float(a) * float(sm)
        top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:topk]
        return top, agg

    top, agg = _run(gate_top_mrna)
    if len(top) == 0 and gate_top_mrna is not None:
        if debug_tag:
            print(f"[WARN] {debug_tag}: gated empty -> fallback ungated")
        top, agg = _run(None)
    return top, agg

# =============================================================================
# Patch image extraction (optional)
# =============================================================================

def _find_wsi_slide_for_patient(pid: str) -> Optional[str]:
    # 常见：WSI_ROOT/pid/*.svs or *.tif
    pdir = os.path.join(WSI_ROOT, pid)
    if not os.path.isdir(pdir):
        return None
    cand = []
    for ext in ["*.svs", "*.tif", "*.tiff", "*.ndpi", "*.mrxs"]:
        cand += glob.glob(os.path.join(pdir, ext))
    cand = sorted(cand)
    return cand[0] if len(cand) else None

def _find_h5_for_patient(pid: str) -> Optional[str]:
    cand = sorted(glob.glob(os.path.join(WSI_H5_ROOT, f"{pid}-*.h5")))
    return cand[0] if len(cand) else None

def _read_coords_from_h5(h5_path: str) -> Optional[np.ndarray]:
    import h5py
    with h5py.File(h5_path, "r") as f:
        for key in ["coords", "coord", "patch_coords", "patch_coord", "xy", "locations"]:
            if key in f and hasattr(f[key], "shape") and len(f[key].shape) == 2 and f[key].shape[1] >= 2:
                arr = f[key][:]
                return arr[:, :2].astype(np.int64)
    return None

def save_top_patches_as_png(pid: str,
                           patch_indices: List[int],
                           out_dir: str,
                           patch_size: int = 256,
                           level: int = 0):
    """
    patch_indices: 这些 index 必须能对应到 coords 的行号（或你的 patch-attn src id）
    """
    try:
        import openslide
    except Exception:
        print("[WARN] openslide-python not available; skip patch image extraction.")
        return

    slide_path = _find_wsi_slide_for_patient(pid)
    h5_path = _find_h5_for_patient(pid)

    if slide_path is None:
        print(f"[WARN] slide not found for {pid} under {WSI_ROOT}")
        return
    if h5_path is None:
        print(f"[WARN] h5 not found for {pid} under {WSI_H5_ROOT}")
        return

    coords = _read_coords_from_h5(h5_path)
    if coords is None:
        print(f"[WARN] coords not found in {h5_path}. Cannot locate patch on slide.")
        return

    slide = openslide.OpenSlide(slide_path)

    safe_mkdir(out_dir)
    for rank, idx in enumerate(patch_indices, start=1):
        if idx < 0 or idx >= coords.shape[0]:
            print(f"[WARN] patch idx out of coords range: {idx} (coordsN={coords.shape[0]})")
            continue
        x, y = int(coords[idx, 0]), int(coords[idx, 1])
        img = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
        out_png = os.path.join(out_dir, f"top{rank}_patchidx{idx}_x{x}_y{y}.png")
        img.save(out_png)

# =============================================================================
# Main
# =============================================================================

def main():
    safe_mkdir(OUT_DIR)
    safe_mkdir(os.path.join(OUT_DIR, "csv"))
    safe_mkdir(os.path.join(OUT_DIR, "patches"))

    pid_a, pid_b, la, lb = get_patient_pair(LABEL_CSV, PATIENT_ID_COL, LABEL_COL, PATIENT_A_ID, PATIENT_B_ID)
    print(f"[PAIR] A={pid_a} (y={la}), B={pid_b} (y={lb})")

    G = load_graph_bin(GRAPH_BIN)
    pid2nid = build_patient_id_to_nid(G, LABEL_CSV, PATIENT_ID_COL)
    if pid_a not in pid2nid or pid_b not in pid2nid:
        raise RuntimeError("Cannot map patient id to nid. Provide graph patient_id strings or correct LABEL_CSV ordering.")
    pa_nid = int(pid2nid[pid_a])
    pb_nid = int(pid2nid[pid_b])
    print(f"[INFO] patient_nid A={pa_nid}, B={pb_nid}")

    name_map_graph = build_name_map_from_graph(G)
    name_map_fallback = build_name_map_fallback_from_omics_csv(OMICS_FILES_FALLBACK)

    num_layers = guess_num_layers(ATTN_DIR)
    print(f"[INFO] detected layers = {num_layers}")
    if num_layers <= 0:
        raise RuntimeError("No attention pt found in ATTN_DIR")

    for layer in range(num_layers):
        layer_dir = os.path.join(OUT_DIR, f"layer_{layer}")
        safe_mkdir(layer_dir)
        csv_layer_dir = os.path.join(OUT_DIR, "csv", f"layer_{layer}")
        safe_mkdir(csv_layer_dir)

        # -------------------------
        # 1) mRNA -> patient gate
        # -------------------------
        f_mrna2p = find_attn_file(ATTN_DIR, layer, *MRNA2PAT)
        if f_mrna2p is None:
            print(f"[MISS] layer {layer}: {MRNA2PAT}")
            continue

        pack_mrna2p = load_attn_pt(f_mrna2p)
        top_mrna_a, mrna_score_a = top_src_for_patient_from_src2patient(pack_mrna2p, pa_nid, topk=TOPK_SAVE_CSV)
        top_mrna_b, mrna_score_b = top_src_for_patient_from_src2patient(pack_mrna2p, pb_nid, topk=TOPK_SAVE_CSV)

        items_plot_a = name_items("gene_mRNA", top_mrna_a[:TOPK_SHOW_PLOT], name_map_graph, name_map_fallback)
        items_plot_b = name_items("gene_mRNA", top_mrna_b[:TOPK_SHOW_PLOT], name_map_graph, name_map_fallback)
        plot_compare_bar(
            title=f"Layer {layer} | Top mRNAs (mRNA→patient)\nA={pid_a}(y={la}) vs B={pid_b}(y={lb})",
            items_a=items_plot_a, items_b=items_plot_b,
            out_path=os.path.join(layer_dir, "mRNA__to__patient__top5.png"),
        )

        # 保存 mRNA gate CSV (Top20)
        top_mrna_a_named = name_items("gene_mRNA", top_mrna_a, name_map_graph, name_map_fallback)[:TOPK_SAVE_CSV]
        top_mrna_b_named = name_items("gene_mRNA", top_mrna_b, name_map_graph, name_map_fallback)[:TOPK_SAVE_CSV]
        save_path_csv(
            out_csv=os.path.join(csv_layer_dir, "PATH_mRNA_to_patient_top20.csv"),
            layer=layer,
            path_name="mRNA→patient (gate)",
            pid_a=pid_a, la=la, pid_b=pid_b, lb=lb,
            top_a=top_mrna_a_named, top_b=top_mrna_b_named
        )

        # gate set（用于上游传播；如果太严会自动 fallback）
        if GATE_TOP_MRNA is None:
            gate_a = None
            gate_b = None
        else:
            gate_a = set([nid for nid, _ in top_mrna_a[:GATE_TOP_MRNA]])
            gate_b = set([nid for nid, _ in top_mrna_b[:GATE_TOP_MRNA]])

        # -------------------------
        # 2) upstream path-weighted
        # -------------------------
        for (src_t, rel, dst_t) in UPSTREAM_RELS:
            f = find_attn_file(ATTN_DIR, layer, src_t, rel, dst_t)
            if f is None:
                print(f"[MISS] layer {layer}: {src_t}__{rel}__{dst_t}")
                continue

            pack = load_attn_pt(f)
            top_a, _ = propagate_upstream_scores(pack, mrna_score_a, topk=TOPK_SAVE_CSV, gate_top_mrna=gate_a,
                                                 debug_tag=f"L{layer} A {src_t}->{dst_t}")
            top_b, _ = propagate_upstream_scores(pack, mrna_score_b, topk=TOPK_SAVE_CSV, gate_top_mrna=gate_b,
                                                 debug_tag=f"L{layer} B {src_t}->{dst_t}")

            # 画图 top5
            items_plot_a = name_items(src_t, top_a[:TOPK_SHOW_PLOT], name_map_graph, name_map_fallback)
            items_plot_b = name_items(src_t, top_b[:TOPK_SHOW_PLOT], name_map_graph, name_map_fallback)
            plot_compare_bar(
                title=f"Layer {layer} | {src_t} --{rel}--> {dst_t}\n(path-weighted by mRNA→patient)\nA={pid_a}(y={la}) vs B={pid_b}(y={lb})",
                items_a=items_plot_a, items_b=items_plot_b,
                out_path=os.path.join(layer_dir, f"{src_t}__{rel}__{dst_t}__path_weighted_top5.png"),
            )

            # 保存 CSV top20
            top_a_named = name_items(src_t, top_a, name_map_graph, name_map_fallback)[:TOPK_SAVE_CSV]
            top_b_named = name_items(src_t, top_b, name_map_graph, name_map_fallback)[:TOPK_SAVE_CSV]
            save_path_csv(
                out_csv=os.path.join(csv_layer_dir, f"PATH_{src_t}__{rel}__{dst_t}_top20.csv"),
                layer=layer,
                path_name=f"{src_t} --{rel}--> {dst_t} (path-weighted)",
                pid_a=pid_a, la=la, pid_b=pid_b, lb=lb,
                top_a=top_a_named, top_b=top_b_named
            )

        # -------------------------
        # 3) patch-level attention -> crop patch images (if exists)
        # -------------------------
        # 需要你有类似：attn_layerX__patch__*__patient.pt
        patch_attn = glob.glob(os.path.join(ATTN_DIR, f"attn_layer{layer}__patch*__*__patient.pt"))
        if len(patch_attn) == 0:
            patch_attn = glob.glob(os.path.join(ATTN_DIR, f"attn_layer{layer}__wsi_patch*__*__patient.pt"))

        if len(patch_attn):
            pf = patch_attn[0]
            pack_patch = load_attn_pt(pf)
            # 这里假设 src=patch_index（对应 h5 coords 行号），dst=patient_nid
            top_patch_a, _ = top_src_for_patient_from_src2patient(pack_patch, pa_nid, topk=TOPK_PATCH)
            top_patch_b, _ = top_src_for_patient_from_src2patient(pack_patch, pb_nid, topk=TOPK_PATCH)

            patch_idx_a = [int(i) for i, _ in top_patch_a]
            patch_idx_b = [int(i) for i, _ in top_patch_b]

            out_pa = os.path.join(OUT_DIR, "patches", f"layer_{layer}", f"A_{pid_a}")
            out_pb = os.path.join(OUT_DIR, "patches", f"layer_{layer}", f"B_{pid_b}")
            save_top_patches_as_png(pid_a, patch_idx_a, out_pa, patch_size=PATCH_SIZE, level=PATCH_LEVEL)
            save_top_patches_as_png(pid_b, patch_idx_b, out_pb, patch_size=PATCH_SIZE, level=PATCH_LEVEL)

            # 同时把 patch idx + score 保存成 csv
            top_patch_a_named = [(f"patch_idx:{int(i)}", float(s)) for i, s in top_patch_a]
            top_patch_b_named = [(f"patch_idx:{int(i)}", float(s)) for i, s in top_patch_b]
            save_path_csv(
                out_csv=os.path.join(csv_layer_dir, f"PATH_patch_to_patient_top{TOPK_PATCH}.csv"),
                layer=layer,
                path_name="patch→patient (top patches)",
                pid_a=pid_a, la=la, pid_b=pid_b, lb=lb,
                top_a=top_patch_a_named, top_b=top_patch_b_named
            )
        else:
            # 如果你只有 wsi->patient（不是 patch），那没法裁 patch 原图
            pass

        print(f"[OK] layer {layer} done -> {layer_dir} / {csv_layer_dir}")

    print("\n[DONE] Outputs saved to:", OUT_DIR)
    print("  - PNG: OUT_DIR/layer_k/*.png")
    print("  - CSV: OUT_DIR/csv/layer_k/*.csv")
    print("  - patches: OUT_DIR/patches/layer_k/*/*.png (if patch-attn + coords available)")

    # 重要提示：如果 gene 名还是像 gene_CNV:123，说明 graph 里没有存字符串；
    # 这时 fallback 会用 *_top.csv 的 gene index 还原。
    print("\n[NOTE] If gene symbols still look like 'gene_CNV:123', your graph probably doesn't store string names.")
    print("       This script falls back to ACC_*_top.csv gene index order, consistent with your graph construction.")


if __name__ == "__main__":
    main()