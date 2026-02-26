# import math
# from typing import Dict, Tuple, Optional, List

# import dgl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.morn_layer import MORNLayer


# class PatientQueryPatchAttn(nn.Module):
#     """
#     Use patient embedding as query to attend over wsi patches.

#     patches: (N, K, D_in)
#     mask:    (N, K) bool/byte (True=valid)
#     query_h: (N, H)

#     Return:
#       wsi_emb: (N, H)
#       attn:    (N, K)
#     """
#     def __init__(self, patch_dim: int, hid_dim: int, dropout: float = 0.1):
#         super().__init__()
#         self.patch_proj = nn.Linear(patch_dim, hid_dim)
#         self.q_proj = nn.Linear(hid_dim, hid_dim)
#         self.k_proj = nn.Linear(hid_dim, hid_dim)
#         self.v_proj = nn.Linear(hid_dim, hid_dim)
#         self.drop = nn.Dropout(dropout)
#         self.scale = math.sqrt(hid_dim)

#     def forward(self, patches: torch.Tensor, mask: Optional[torch.Tensor], query_h: torch.Tensor):
#         """
#         patches: (N,K,D)
#         mask: (N,K) bool or uint8/byte. True means valid patch.
#         query_h: (N,H)
#         """
#         # project patches to hid
#         p = F.gelu(self.patch_proj(patches))       # (N,K,H)
#         q = self.q_proj(query_h).unsqueeze(1)      # (N,1,H)
#         k = self.k_proj(p)                         # (N,K,H)
#         v = self.v_proj(p)                         # (N,K,H)

#         # (N,K)
#         score = (q * k).sum(dim=-1).squeeze(1) / self.scale

#         if mask is not None:
#             # ✅ ensure boolean mask + correct device
#             if mask.dtype != torch.bool:
#                 mask = mask.to(torch.bool)
#             mask = mask.to(device=score.device)
#             score = score.masked_fill(~mask, float("-inf"))

#         attn = torch.softmax(score, dim=1)         # (N,K)
#         attn = self.drop(attn)

#         wsi_emb = torch.bmm(attn.unsqueeze(1), v).squeeze(1)  # (N,H)
#         return wsi_emb, attn


# class MORN(nn.Module):
#     """
#     Two-stage:
#       A) omics-only HGT (exclude any etypes involving 'wsi')
#       B) patient->patch cross-attn to get wsi embedding
#       C) cross HGT on full graph for omics<->wsi interaction
#     """
#     def __init__(
#         self,
#         G: dgl.DGLHeteroGraph,
#         node_dict: Dict[str, int],
#         edge_dict: Dict[Tuple[str, str, str], int],
#         n_hid: int,
#         n_out: int,
#         n_layers_omics: int = 2,
#         n_layers_cross: int = 1,
#         n_heads: int = 4,
#         use_norm: bool = True,
#         wsi_patch_dim: int = 1024,
#         edge_weight_key: str = "w",
#         edge_weight_mode: str = "mul_attn",
#         dropout: float = 0.2,
#     ):
#         super().__init__()
#         self.node_dict = node_dict
#         self.edge_dict = edge_dict
#         self.n_hid = n_hid

#         # trainable embeddings for all non-wsi types
#         self.emb = nn.ModuleDict()
#         for ntype in G.ntypes:
#             if ntype == "wsi":
#                 continue
#             self.emb[ntype] = nn.Embedding(G.num_nodes(ntype), n_hid)

#         # patch attention (patient-query)
#         self.patch_attn = PatientQueryPatchAttn(
#             patch_dim=wsi_patch_dim,
#             hid_dim=n_hid,
#             dropout=dropout,
#         )

#         # omics-only etypes (exclude any involving 'wsi')
#         self.omics_canonical_etypes: List[Tuple[str, str, str]] = [
#             et for et in G.canonical_etypes if ("wsi" not in et)
#         ]

#         self.layers_omics = nn.ModuleList([
#             MORNLayer(
#                 in_dim=n_hid,
#                 out_dim=n_hid,
#                 node_dict=node_dict,
#                 edge_dict=edge_dict,
#                 n_heads=n_heads,
#                 dropout=dropout,
#                 use_norm=use_norm,
#                 edge_weight_key=edge_weight_key,
#                 edge_weight_mode=edge_weight_mode,
#             )
#             for _ in range(n_layers_omics)
#         ])

#         self.layers_cross = nn.ModuleList([
#             MORNLayer(
#                 in_dim=n_hid,
#                 out_dim=n_hid,
#                 node_dict=node_dict,
#                 edge_dict=edge_dict,
#                 n_heads=n_heads,
#                 dropout=dropout,
#                 use_norm=use_norm,
#                 edge_weight_key=edge_weight_key,
#                 edge_weight_mode=edge_weight_mode,
#             )
#             for _ in range(n_layers_cross)
#         ])

#         self.out = nn.Linear(n_hid, n_out)

#     def _init_h_nonwsi(self, G: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
#         h = {}
#         for ntype in G.ntypes:
#             if ntype == "wsi":
#                 continue
#             nid = G.nodes[ntype].data["nid"]
#             h[ntype] = self.emb[ntype](nid)
#         return h

#     def forward(
#         self,
#         G: dgl.DGLHeteroGraph,
#         out_key: str = "patient",
#         return_attn: bool = False,
#         return_patch_attn: bool = False,
#     ):
#         all_edge_attn = [] if return_attn else None

#         # ---------- Stage A: omics-only ----------
#         h = self._init_h_nonwsi(G)
#         G_omics = dgl.edge_type_subgraph(G, self.omics_canonical_etypes)

#         for layer in self.layers_omics:
#             if return_attn:
#                 h, attn_dict = layer(G_omics, h, return_attn=True)
#                 all_edge_attn.append(attn_dict)
#             else:
#                 h = layer(G_omics, h, return_attn=False)

#         # patient embedding as query
#         patient_h = h["patient"]  # (N,H)

#         patches = G.nodes["wsi"].data["wsi_patches"]  # (N,K,D)
#         mask = G.nodes["wsi"].data.get("wsi_patch_mask", None)  # (N,K) bool/byte

#         wsi_h, patch_attn = self.patch_attn(patches, mask, patient_h)  # (N,H), (N,K)

#         # ---------- Stage B: cross interaction on full graph ----------
#         h_full = dict(h)
#         h_full["wsi"] = wsi_h

#         for layer in self.layers_cross:
#             if return_attn:
#                 h_full, attn_dict = layer(G, h_full, return_attn=True)
#                 all_edge_attn.append(attn_dict)
#             else:
#                 h_full = layer(G, h_full, return_attn=False)

#         logits = self.out(h_full[out_key])

#         if return_attn and return_patch_attn:
#             return logits, all_edge_attn, patch_attn
#         if return_attn:
#             return logits, all_edge_attn
#         if return_patch_attn:
#             return logits, patch_attn
#         return logits


# import math
# from typing import Dict, Tuple, Optional, List

# import dgl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.morn_layer import MORNLayer


# class PatientQueryPatchAttn(nn.Module):
#     """
#     Use patient embedding as query to attend over wsi patches.

#     patches: (N, K, D_in)
#     mask:    (N, K) bool (True=valid)
#     query_h: (N, H)

#     Return:
#       wsi_emb: (N, H)
#       attn:    (N, K)
#     """
#     def __init__(self, patch_dim: int, hid_dim: int, dropout: float = 0.1):
#         super().__init__()
#         self.patch_proj = nn.Linear(patch_dim, hid_dim)
#         self.q_proj = nn.Linear(hid_dim, hid_dim)
#         self.k_proj = nn.Linear(hid_dim, hid_dim)
#         self.v_proj = nn.Linear(hid_dim, hid_dim)
#         self.drop = nn.Dropout(dropout)
#         self.scale = math.sqrt(hid_dim)

#     def forward(self, patches: torch.Tensor, mask: Optional[torch.Tensor], query_h: torch.Tensor):
#         # project patches to hid
#         p = F.gelu(self.patch_proj(patches))       # (N,K,H)
#         q = self.q_proj(query_h).unsqueeze(1)      # (N,1,H)
#         k = self.k_proj(p)                         # (N,K,H)
#         v = self.v_proj(p)                         # (N,K,H)

#         # (N,K)
#         score = (q * k).sum(dim=-1).squeeze(1) / self.scale

#         if mask is not None:
#             if mask.dtype != torch.bool:
#                 mask = mask.to(torch.bool)
#             mask = mask.to(device=score.device)
#             score = score.masked_fill(~mask, float("-inf"))

#         attn = torch.softmax(score, dim=1)         # (N,K)
#         attn = self.drop(attn)

#         wsi_emb = torch.bmm(attn.unsqueeze(1), v).squeeze(1)  # (N,H)
#         return wsi_emb, attn


# class MORN(nn.Module):
#     """
#     Two-stage:
#       A) omics-only HGT (exclude any etypes involving 'wsi')
#       B) patient->patch cross-attn to get wsi embedding
#       C) cross HGT on full graph for omics<->wsi interaction
#     """
#     def __init__(
#         self,
#         G: dgl.DGLHeteroGraph,
#         node_dict: Dict[str, int],
#         edge_dict: Dict[Tuple[str, str, str], int],
#         n_hid: int,
#         n_out: int,
#         n_layers_omics: int = 2,
#         n_layers_cross: int = 1,
#         n_heads: int = 4,
#         use_norm: bool = True,
#         wsi_patch_dim: int = 1024,
#         edge_weight_key: str = "w",
#         edge_weight_mode: str = "mul_attn",
#         dropout: float = 0.2,
#     ):
#         super().__init__()
#         self.node_dict = node_dict
#         self.edge_dict = edge_dict
#         self.n_hid = n_hid
#         self.edge_weight_key = edge_weight_key
#         self.edge_weight_mode = edge_weight_mode

#         # trainable embeddings for all non-wsi types
#         self.emb = nn.ModuleDict()
#         for ntype in G.ntypes:
#             if ntype == "wsi":
#                 continue
#             self.emb[ntype] = nn.Embedding(G.num_nodes(ntype), n_hid)

#         # patch attention (patient-query)
#         self.patch_attn = PatientQueryPatchAttn(
#             patch_dim=wsi_patch_dim,
#             hid_dim=n_hid,
#             dropout=dropout,
#         )

#         # omics-only etypes (exclude any involving 'wsi')
#         self.omics_canonical_etypes: List[Tuple[str, str, str]] = [
#             et for et in G.canonical_etypes if ("wsi" not in et)
#         ]

#         self.layers_omics = nn.ModuleList([
#             MORNLayer(
#                 in_dim=n_hid,
#                 out_dim=n_hid,
#                 node_dict=node_dict,
#                 edge_dict=edge_dict,
#                 n_heads=n_heads,
#                 dropout=dropout,
#                 use_norm=use_norm,
#                 edge_weight_key=edge_weight_key,
#                 edge_weight_mode=edge_weight_mode,
#             )
#             for _ in range(n_layers_omics)
#         ])

#         self.layers_cross = nn.ModuleList([
#             MORNLayer(
#                 in_dim=n_hid,
#                 out_dim=n_hid,
#                 node_dict=node_dict,
#                 edge_dict=edge_dict,
#                 n_heads=n_heads,
#                 dropout=dropout,
#                 use_norm=use_norm,
#                 edge_weight_key=edge_weight_key,
#                 edge_weight_mode=edge_weight_mode,
#             )
#             for _ in range(n_layers_cross)
#         ])

#         self.out = nn.Linear(n_hid, n_out)

#         # in models/morn.py  (inside __init__)
#         self.wsi_proj = torch.nn.Linear(wsi_patch_dim, n_hid)


#     def _init_h_nonwsi(self, G: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
#         h = {}
#         for ntype in G.ntypes:
#             if ntype == "wsi":
#                 continue
#             nid = G.nodes[ntype].data["nid"]
#             h[ntype] = self.emb[ntype](nid)
#         return h

#     def forward(
#         self,
#         G: dgl.DGLHeteroGraph,
#         out_key: str = "patient",
#         return_attn: bool = False,
#         return_patch_attn: bool = False,
#         return_omics_h: bool = False,   # ✅ 新增：返回 Stage-A gene embeddings for contrastive
#     ):
#         all_edge_attn = [] if return_attn else None

#         # ---------- Stage A: omics-only ----------
#         h = self._init_h_nonwsi(G)
#         G_omics = dgl.edge_type_subgraph(G, self.omics_canonical_etypes)

#         for layer in self.layers_omics:
#             if return_attn:
#                 h, attn_dict = layer(G_omics, h, return_attn=True)
#                 all_edge_attn.append(attn_dict)
#             else:
#                 h = layer(G_omics, h, return_attn=False)

#         # keep a copy of omics-only embeddings (after Stage-A)
#         omics_h = None
#         if return_omics_h:
#             # include only gene_* types (and patient if you want)
#             omics_h = {k: v for k, v in h.items() if k.startswith("gene_")}

#         # patient embedding as query
#         patient_h = h["patient"]  # (N,H)

#         patches = G.nodes["wsi"].data["wsi_patches"]  # (N,K,D)
#         mask = G.nodes["wsi"].data.get("wsi_patch_mask", None)  # (N,K) bool/byte

#         wsi_h, patch_attn = self.patch_attn(patches, mask, patient_h)  # (N,H), (N,K)

#         # patient_h = h["patient"]  # (N,H)

#         # patches = G.nodes["wsi"].data["wsi_patches"]  # (N,K,D)
#         # mask = G.nodes["wsi"].data.get("wsi_patch_mask", None)  # (N,K) bool/byte

#         # # ---- mean pooling over patches (with optional mask) ----
#         # if mask is None:
#         #     # (N,K,D) -> (N,D)
#         #     pooled = patches.mean(dim=1)
#         # else:
#         #     # ensure bool
#         #     m = mask.bool()  # (N,K)
#         #     # (N,K,1)
#         #     m3 = m.unsqueeze(-1).to(patches.dtype)
#         #     # masked sum / count
#         #     denom = m3.sum(dim=1).clamp_min(1.0)  # (N,1)
#         #     pooled = (patches * m3).sum(dim=1) / denom  # (N,D)

#         # # project to hidden dim H
#         # wsi_h = self.wsi_proj(pooled)  # (N,H)

#         # # no attention now; keep a placeholder for compatibility
#         # patch_attn = None


#         # ---------- Stage B: cross interaction on full graph ----------
        
#         h_full = dict(h)
#         h_full["wsi"] = wsi_h

#         for layer in self.layers_cross:
#             if return_attn:
#                 h_full, attn_dict = layer(G, h_full, return_attn=True)
#                 all_edge_attn.append(attn_dict)
#             else:
#                 h_full = layer(G, h_full, return_attn=False)

#         logits = self.out(h_full[out_key])

#         # ---- returns ----
#         # 组合返回：尽量保持你原来的用法不变
#         if return_attn and return_patch_attn and return_omics_h:
#             return logits, all_edge_attn, patch_attn, omics_h
#         if return_attn and return_patch_attn:
#             return logits, all_edge_attn, patch_attn
#         if return_attn and return_omics_h:
#             return logits, all_edge_attn, omics_h
#         if return_patch_attn and return_omics_h:
#             return logits, patch_attn, omics_h
#         if return_attn:
#             return logits, all_edge_attn
#         if return_patch_attn:
#             return logits, patch_attn
#         if return_omics_h:
#             return logits, omics_h
#         return logits


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Dict, Tuple, Optional, List, Any

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.morn_layer import MORNLayer


class PatientQueryPatchAttn(nn.Module):
    """
    Use patient embedding as query to attend over wsi patches.

    patches: (N, K, D_in)
    mask:    (N, K) bool (True=valid)
    query_h: (N, H)

    Return:
      wsi_emb: (N, H)
      attn:    (N, K)
    """
    def __init__(self, patch_dim: int, hid_dim: int, dropout: float = 0.1):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, hid_dim)
        self.q_proj = nn.Linear(hid_dim, hid_dim)
        self.k_proj = nn.Linear(hid_dim, hid_dim)
        self.v_proj = nn.Linear(hid_dim, hid_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim)

    def forward(self, patches: torch.Tensor, mask: Optional[torch.Tensor], query_h: torch.Tensor):
        # project patches to hid
        p = F.gelu(self.patch_proj(patches))       # (N,K,H)
        q = self.q_proj(query_h).unsqueeze(1)      # (N,1,H)
        k = self.k_proj(p)                         # (N,K,H)
        v = self.v_proj(p)                         # (N,K,H)

        # (N,K)
        score = (q * k).sum(dim=-1).squeeze(1) / self.scale

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            mask = mask.to(device=score.device)
            score = score.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(score, dim=1)         # (N,K)
        attn = self.drop(attn)

        wsi_emb = torch.bmm(attn.unsqueeze(1), v).squeeze(1)  # (N,H)
        return wsi_emb, attn


class MORN(nn.Module):
    """
    Two-stage (optional WSI):
      A) omics-only HGT (exclude any etypes involving 'wsi')
      B) patient->patch cross-attn to get wsi embedding (if use_wsi & graph has 'wsi')
      C) cross HGT on full graph for omics<->wsi interaction (if use_wsi & graph has 'wsi')
         otherwise just run omics-only output head on patient.
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
        n_heads: int = 4,
        use_norm: bool = True,
        wsi_patch_dim: int = 1024,
        edge_weight_key: str = "w",
        edge_weight_mode: str = "mul_attn",
        dropout: float = 0.2,
        use_wsi: bool = True,  # ✅ 新增：允许禁用WSI分支
    ):
        super().__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_hid = n_hid

        self.edge_weight_key = edge_weight_key
        self.edge_weight_mode = edge_weight_mode

        # ✅ WSI是否可用：既要 use_wsi=True 也要图里确实有wsi ntype
        self.use_wsi = bool(use_wsi) and ("wsi" in G.ntypes)

        # trainable embeddings for all non-wsi types
        self.emb = nn.ModuleDict()
        for ntype in G.ntypes:
            if ntype == "wsi":
                continue
            self.emb[ntype] = nn.Embedding(G.num_nodes(ntype), n_hid)

        # patch attention (patient-query) -- only meaningful if WSI exists
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
            MORNLayer(
                in_dim=n_hid,
                out_dim=n_hid,
                node_dict=node_dict,
                edge_dict=edge_dict,
                n_heads=n_heads,
                dropout=dropout,
                use_norm=use_norm,
                edge_weight_key=edge_weight_key,
                edge_weight_mode=edge_weight_mode,
            )
            for _ in range(int(n_layers_omics))
        ])

        self.layers_cross = nn.ModuleList([
            MORNLayer(
                in_dim=n_hid,
                out_dim=n_hid,
                node_dict=node_dict,
                edge_dict=edge_dict,
                n_heads=n_heads,
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
        return_omics_h: bool = False,   # ✅ 返回 Stage-A gene embeddings for contrastive
    ):
        all_edge_attn = [] if return_attn else None

        # ---------- Stage A: omics-only ----------
        h = self._init_h_nonwsi(G)

        # 如果没有任何 omics etype（比如你只保留 patient<->wsi），那就跳过 omics-HGT
        if len(self.omics_canonical_etypes) > 0 and len(self.layers_omics) > 0:
            G_omics = dgl.edge_type_subgraph(G, self.omics_canonical_etypes)
            for layer in self.layers_omics:
                if return_attn:
                    h, attn_dict = layer(G_omics, h, return_attn=True)
                    all_edge_attn.append(attn_dict)
                else:
                    h = layer(G_omics, h, return_attn=False)

        # keep a copy of omics-only embeddings (after Stage-A)
        omics_h = None
        if return_omics_h:
            omics_h = {k: v for k, v in h.items() if k.startswith("gene_")}

        patch_attn = None

        # ---------- Stage B/C: optional WSI cross ----------
        if self.use_wsi:
            # patient embedding as query
            patient_h = h["patient"]  # (N,H)

            patches = G.nodes["wsi"].data["wsi_patches"]  # (N,K,D)
            mask = G.nodes["wsi"].data.get("wsi_patch_mask", None)  # (N,K)

            wsi_h, patch_attn = self.patch_attn(patches, mask, patient_h)  # (N,H), (N,K)

            # cross interaction on full graph
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
            # no WSI branch: directly output from omics-only patient representation
            logits = self.out(h[out_key])

        # ---- returns (保持你原来的用法不变) ----
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
