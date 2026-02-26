#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Dict, Tuple, Optional, List

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.functional import edge_softmax


# ============================================================
# RGCN-style hetero layer (canonical-safe) + "attn-like" outputs
# ============================================================
class MORNLayerRGCN(nn.Module):
    """
    RGCN-style message passing on heterograph:
      - per canonical relation has its own Linear (W_r)
      - message = W_r h_src
      - optionally multiply by edge weight edata[w]
      - aggregate to dst by sum
      - cross-reducer across relations uses "mean" (like your HGT code)
    return_attn:
      - returns per-edge normalized weights (E,1) for each canonical etype
        (if edge weights exist, normalize them by dst using edge_softmax)
        (otherwise uniform by dst using edge_softmax(zeros))
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        node_dict: Dict[str, int],
        edge_dict: Dict[Tuple[str, str, str], int],
        dropout: float = 0.2,
        use_norm: bool = False,
        edge_weight_key: str = "w",
        edge_weight_mode: str = "mul_msg",  # "mul_msg" or "none"
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.use_norm = bool(use_norm)

        self.edge_weight_key = edge_weight_key
        self.edge_weight_mode = edge_weight_mode

        # type-specific output transform (like HGT's a_linears)
        self.a_linears = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(len(node_dict))])
        self.norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(len(node_dict))]) if use_norm else None

        # relation-specific linear (canonical-safe)
        # use edge_dict keys as canonical relations
        self.rel_linears = nn.ModuleDict({
            self._rel_key(c_etype): nn.Linear(in_dim, out_dim)
            for c_etype in edge_dict.keys()
        })

        # skip for each node type
        self.skip = nn.Parameter(torch.ones(len(node_dict)))
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _rel_key(c_etype: Tuple[str, str, str]) -> str:
        return f"{c_etype[0]}__{c_etype[1]}__{c_etype[2]}"

    def forward(
        self,
        G: dgl.DGLHeteroGraph,
        h: Dict[str, torch.Tensor],
        return_attn: bool = False,
    ):
        attn_dict = {} if return_attn else None

        with G.local_scope():
            funcs = {}

            for c_etype in G.canonical_etypes:
                if c_etype not in self.edge_dict:
                    continue

                sub = G[c_etype]
                srctype, etype, dsttype = c_etype
                key = self._rel_key(c_etype)

                # (Ns, out_dim)
                msg = self.rel_linears[key](h[srctype])
                sub.srcdata["m_src"] = msg

                # edge weight handling
                if self.edge_weight_mode == "mul_msg" and (self.edge_weight_key in sub.edata):
                    w = sub.edata[self.edge_weight_key].float()

                    # build "attn-like" normalized edge weight for visualization (E,1)
                    if return_attn:
                        # normalize by dst
                        # edge_softmax expects (E,) or (E,1). Use (E,1).
                        a = edge_softmax(sub, w.unsqueeze(-1), norm_by="dst")  # (E,1)
                        attn_dict[c_etype] = a.detach().cpu()
                    # use raw w for message scaling (not normalized) or you can use normalized a
                    sub.edata["ew"] = w.unsqueeze(-1)  # (E,1)
                    funcs[c_etype] = (
                        fn.u_mul_e("m_src", "ew", "m"),
                        fn.sum("m", "t"),
                    )
                else:
                    # uniform "attn-like" weights if requested
                    if return_attn:
                        zeros = torch.zeros(sub.num_edges(), 1, device=h[srctype].device)
                        a = edge_softmax(sub, zeros, norm_by="dst")  # uniform per dst
                        attn_dict[c_etype] = a.detach().cpu()

                    funcs[c_etype] = (
                        fn.copy_u("m_src", "m"),
                        fn.sum("m", "t"),
                    )

            if len(funcs) > 0:
                # cross_reducer="mean" 保持与你 HGT 版本一致（避免关系多的类型被 sum 放大）
                G.multi_update_all(funcs, cross_reducer="mean")
            else:
                # no edges -> nothing to aggregate
                pass

            new_h = {}
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])

                if "t" in G.nodes[ntype].data:
                    t = G.nodes[ntype].data["t"]
                else:
                    t = torch.zeros(G.num_nodes(ntype), self.out_dim, device=h[ntype].device)

                trans = self.drop(self.a_linears[n_id](t))
                out = trans * alpha + h[ntype] * (1.0 - alpha)
                if self.use_norm:
                    out = self.norms[n_id](out)
                new_h[ntype] = out

            if return_attn:
                return new_h, attn_dict
            return new_h


# ============================================================
# Patient -> WSI patches attention (unchanged)
# ============================================================
class PatientQueryPatchAttn(nn.Module):
    def __init__(self, patch_dim: int, hid_dim: int, dropout: float = 0.1):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, hid_dim)
        self.q_proj = nn.Linear(hid_dim, hid_dim)
        self.k_proj = nn.Linear(hid_dim, hid_dim)
        self.v_proj = nn.Linear(hid_dim, hid_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim)

    def forward(self, patches: torch.Tensor, mask: Optional[torch.Tensor], query_h: torch.Tensor):
        p = F.gelu(self.patch_proj(patches))       # (N,K,H)
        q = self.q_proj(query_h).unsqueeze(1)      # (N,1,H)
        k = self.k_proj(p)                         # (N,K,H)
        v = self.v_proj(p)                         # (N,K,H)

        score = (q * k).sum(dim=-1).squeeze(1) / self.scale  # (N,K)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            mask = mask.to(device=score.device)
            score = score.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(score, dim=1)         # (N,K)
        attn = self.drop(attn)

        wsi_emb = torch.bmm(attn.unsqueeze(1), v).squeeze(1)  # (N,H)
        return wsi_emb, attn


# ============================================================
# MORN (RGCN backbone)
# ============================================================
class MORN(nn.Module):
    """
    Two-stage (optional WSI):
      A) omics-only RGCN on subgraph (exclude any etypes involving 'wsi')
      B) patient->patch attention to get wsi embedding (if use_wsi)
      C) cross RGCN on full graph (if use_wsi) otherwise directly head on patient
    """
    def __init__(
        self,
        G: dgl.DGLHeteroGraph,
        node_dict: Dict[str, int],
        edge_dict: Dict[Tuple[str, str, str], int],
        n_hid: int,
        n_out: int,
        n_layers_omics: int = 2,
        n_layers_cross: int = 1,
        use_norm: bool = True,
        wsi_patch_dim: int = 1024,
        edge_weight_key: str = "w",
        edge_weight_mode: str = "mul_msg",  # for RGCN branch
        dropout: float = 0.2,
        use_wsi: bool = True,
    ):
        super().__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_hid = int(n_hid)

        self.edge_weight_key = edge_weight_key
        self.edge_weight_mode = edge_weight_mode

        self.use_wsi = bool(use_wsi) and ("wsi" in G.ntypes)

        # trainable embeddings for all non-wsi types
        self.emb = nn.ModuleDict()
        for ntype in G.ntypes:
            if ntype == "wsi":
                continue
            self.emb[ntype] = nn.Embedding(G.num_nodes(ntype), n_hid)

        # patch attention
        self.patch_attn = PatientQueryPatchAttn(
            patch_dim=wsi_patch_dim,
            hid_dim=n_hid,
            dropout=dropout,
        )

        # omics-only etypes (exclude any involving 'wsi')
        self.omics_canonical_etypes: List[Tuple[str, str, str]] = [
            et for et in G.canonical_etypes if ("wsi" not in et)
        ]

        self.layers_omics = nn.ModuleList([
            MORNLayerRGCN(
                in_dim=n_hid,
                out_dim=n_hid,
                node_dict=node_dict,
                edge_dict=edge_dict,
                dropout=dropout,
                use_norm=use_norm,
                edge_weight_key=edge_weight_key,
                edge_weight_mode=edge_weight_mode,
            )
            for _ in range(int(n_layers_omics))
        ])

        self.layers_cross = nn.ModuleList([
            MORNLayerRGCN(
                in_dim=n_hid,
                out_dim=n_hid,
                node_dict=node_dict,
                edge_dict=edge_dict,
                dropout=dropout,
                use_norm=use_norm,
                edge_weight_key=edge_weight_key,
                edge_weight_mode=edge_weight_mode,
            )
            for _ in range(int(n_layers_cross))
        ])

        self.out = nn.Linear(n_hid, n_out)

    def _init_h_nonwsi(self, G: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
        h = {}
        for ntype in G.ntypes:
            if ntype == "wsi":
                continue
            nid = G.nodes[ntype].data["nid"]
            h[ntype] = self.emb[ntype](nid)
        return h

    def forward(
        self,
        G: dgl.DGLHeteroGraph,
        out_key: str = "patient",
        return_attn: bool = False,
        return_patch_attn: bool = False,
        return_omics_h: bool = False,
    ):
        all_edge_attn = [] if return_attn else None

        # ---------- Stage A: omics-only ----------
        h = self._init_h_nonwsi(G)

        if len(self.omics_canonical_etypes) > 0 and len(self.layers_omics) > 0:
            G_omics = dgl.edge_type_subgraph(G, self.omics_canonical_etypes)
            for layer in self.layers_omics:
                if return_attn:
                    h, attn_dict = layer(G_omics, h, return_attn=True)
                    all_edge_attn.append(attn_dict)
                else:
                    h = layer(G_omics, h, return_attn=False)

        omics_h = None
        if return_omics_h:
            omics_h = {k: v for k, v in h.items() if k.startswith("gene_")}

        patch_attn = None

        # ---------- Stage B/C: optional WSI ----------
        if self.use_wsi:
            patient_h = h["patient"]

            patches = G.nodes["wsi"].data["wsi_patches"]  # (N,K,D)
            mask = G.nodes["wsi"].data.get("wsi_patch_mask", None)

            wsi_h, patch_attn = self.patch_attn(patches, mask, patient_h)

            h_full = dict(h)
            h_full["wsi"] = wsi_h

            if len(self.layers_cross) > 0:
                for layer in self.layers_cross:
                    if return_attn:
                        h_full, attn_dict = layer(G, h_full, return_attn=True)
                        all_edge_attn.append(attn_dict)
                    else:
                        h_full = layer(G, h_full, return_attn=False)

            logits = self.out(h_full[out_key])
        else:
            logits = self.out(h[out_key])

        # ---- returns (保持接口一致) ----
        if return_attn and return_patch_attn and return_omics_h:
            return logits, all_edge_attn, patch_attn, omics_h
        if return_attn and return_patch_attn:
            return logits, all_edge_attn, patch_attn
        if return_attn and return_omics_h:
            return logits, all_edge_attn, omics_h
        if return_patch_attn and return_omics_h:
            return logits, patch_attn, omics_h
        if return_attn:
            return logits, all_edge_attn
        if return_patch_attn:
            return logits, patch_attn
        if return_omics_h:
            return logits, omics_h
        return logits
