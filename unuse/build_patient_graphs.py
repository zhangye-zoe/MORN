#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, Tuple, List, Optional

import torch
import dgl


def ensure_nid(G: dgl.DGLHeteroGraph):
    """Ensure each node type has nid feature (global id)."""
    for ntype in G.ntypes:
        if "nid" not in G.nodes[ntype].data:
            G.nodes[ntype].data["nid"] = torch.arange(G.num_nodes(ntype), dtype=torch.long)


def pick_patients_from_meta(meta_json: str) -> List[str]:
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    patients = meta.get("patients", None)
    if patients is None:
        raise KeyError(f"meta json missing 'patients': {meta_json}")
    return patients


def _safe_cat_unique(ids: List[torch.Tensor]) -> torch.Tensor:
    if len(ids) == 0:
        return torch.empty(0, dtype=torch.long)
    x = torch.cat(ids, dim=0)
    return torch.unique(x)


def build_one_patient_graph(
    G: dgl.DGLHeteroGraph,
    patient_ntype: str,
    patient_id: int,
    use_wsi: bool = True,
    use_modal: Dict[str, bool] = None,
    keep_regulation: bool = True,
) -> dgl.DGLHeteroGraph:
    """
    Build per-patient heterograph:
      - keep exactly 1 patient node (local)
      - keep its 1-hop neighbors in selected modalities
      - keep regulation edges only among kept omics nodes (if keep_regulation)

    Assumptions:
      - patient nodes are indexed in G.nodes[patient_ntype] by patient_id (consistent with splits/meta)
      - omics nodes types could be: gene_CNV, gene_Methy, gene_mRNA, gene_miRNA (customize below)
      - WSI node type is "wsi" (optional)
    """
    if use_modal is None:
        use_modal = {
            "gene_CNV": True,
            "gene_Methy": True,
            "gene_mRNA": True,
            "gene_miRNA": True,
        }

    # 1) Start with the single patient node
    seed = {patient_ntype: torch.tensor([patient_id], dtype=torch.long)}

    # 2) Collect 1-hop neighbors by traversing all canonical etypes
    #    but restrict to modalities and optionally wsi
    keep_nodes: Dict[str, torch.Tensor] = {patient_ntype: seed[patient_ntype]}
    for ntype in G.ntypes:
        if ntype == patient_ntype:
            continue
        if ntype == "wsi" and not use_wsi:
            continue
        if ntype != "wsi" and ntype.startswith("gene_") and (not use_modal.get(ntype, True)):
            continue
        # default: allow other types if exist
        keep_nodes.setdefault(ntype, torch.empty(0, dtype=torch.long))

    # Scan edges incident to this patient node
    # We include both directions: patient->X and X->patient
    for (srctype, etype, dsttype) in G.canonical_etypes:
        if (srctype == patient_ntype and dsttype in keep_nodes):
            # patient -> dsttype neighbors
            u, v = G.edges(etype=(srctype, etype, dsttype))
            mask = (u == patient_id)
            nb = v[mask]
            if nb.numel() > 0:
                keep_nodes[dsttype] = torch.unique(torch.cat([keep_nodes[dsttype], nb]))
        if (dsttype == patient_ntype and srctype in keep_nodes):
            # srctype -> patient neighbors
            u, v = G.edges(etype=(srctype, etype, dsttype))
            mask = (v == patient_id)
            nb = u[mask]
            if nb.numel() > 0:
                keep_nodes[srctype] = torch.unique(torch.cat([keep_nodes[srctype], nb]))

    # remove empty ntypes that are not present
    keep_nodes = {k: v for k, v in keep_nodes.items() if v.numel() > 0}

    # 3) Node-induced subgraph (preserves edges among kept nodes)
    #    This will keep regulation edges among omics nodes IF both endpoints kept.
    Gp = dgl.node_subgraph(G, keep_nodes, store_ids=True)

    # 4) Optionally drop regulation edges if user wants
    #    (regulation edges are those not incident to patient/wsi)
    if not keep_regulation:
        keep_etypes = []
        for et in Gp.canonical_etypes:
            s, r, d = et
            if patient_ntype in (s, d):
                keep_etypes.append(et)
            elif "wsi" in (s, d):
                keep_etypes.append(et)
        if len(keep_etypes) == 0:
            raise RuntimeError("After dropping regulation edges, no edges left in patient graph.")
        Gp = dgl.edge_type_subgraph(Gp, keep_etypes, preserve_nodes=True)

    return Gp


def main():
    ap = argparse.ArgumentParser("Build per-patient graphs from a global heterograph")
    ap.add_argument("--graph_bin", type=str, required=True, help="path to <DATASET>_graph.bin")
    ap.add_argument("--meta_json", type=str, required=True, help="meta json containing patients list")
    ap.add_argument("--out_bin", type=str, required=True, help="output patient_graphs.bin")
    ap.add_argument("--out_meta", type=str, required=True, help="output patient_graphs_meta.json")
    ap.add_argument("--patient_ntype", type=str, default="patient")
    ap.add_argument("--use_wsi", type=int, default=1, choices=[0, 1])
    ap.add_argument("--keep_regulation", type=int, default=1, choices=[0, 1])
    # modality switches
    ap.add_argument("--use_cnv", type=int, default=1, choices=[0, 1])
    ap.add_argument("--use_methy", type=int, default=1, choices=[0, 1])
    ap.add_argument("--use_mrna", type=int, default=1, choices=[0, 1])
    ap.add_argument("--use_mirna", type=int, default=1, choices=[0, 1])

    args = ap.parse_args()

    graphs, _ = dgl.load_graphs(args.graph_bin)
    G = graphs[0]
    ensure_nid(G)

    patients = pick_patients_from_meta(args.meta_json)
    # important: patient_id in global graph is index in this patients list order (your pipeline uses it)
    # If your patient node ordering differs, you must provide a mapping here.
    # We assume patient_id == position in patients list.
    print(f"[INFO] total patients in meta: {len(patients)}")

    use_modal = {
        "gene_CNV": bool(args.use_cnv),
        "gene_Methy": bool(args.use_methy),
        "gene_mRNA": bool(args.use_mrna),
        "gene_miRNA": bool(args.use_mirna),
    }

    out_graphs = []
    out_info = []

    for pid in range(len(patients)):
        gpid = build_one_patient_graph(
            G=G,
            patient_ntype=args.patient_ntype,
            patient_id=pid,
            use_wsi=bool(args.use_wsi),
            use_modal=use_modal,
            keep_regulation=bool(args.keep_regulation),
        )
        out_graphs.append(gpid)
        out_info.append({
            "patient_index": pid,
            "patient_id": patients[pid],
            "num_nodes": {nt: int(gpid.num_nodes(nt)) for nt in gpid.ntypes},
            "num_edges": {str(et): int(gpid.num_edges(et)) for et in gpid.canonical_etypes},
        })

        if (pid + 1) % 100 == 0:
            print(f"[BUILD] {pid+1}/{len(patients)}")

    os.makedirs(os.path.dirname(args.out_bin), exist_ok=True)
    dgl.save_graphs(args.out_bin, out_graphs)
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "graph_bin": args.graph_bin,
            "meta_json": args.meta_json,
            "patient_ntype": args.patient_ntype,
            "use_wsi": bool(args.use_wsi),
            "keep_regulation": bool(args.keep_regulation),
            "use_modal": use_modal,
            "patients": patients,
            "per_patient": out_info,
        }, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {args.out_bin}")
    print(f"[SAVED] {args.out_meta}")


if __name__ == "__main__":
    main()
