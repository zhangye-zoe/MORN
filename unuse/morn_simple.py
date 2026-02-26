import math
from typing import Dict, Tuple, Optional, List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.morn_layer import MORNLayer


def masked_mean_pool(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: (N,K,D)
    mask: (N,K) bool, True=valid
    return: (N,D)
    """
    if mask is None:
        return x.mean(dim=1)

    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    mask = mask.to(device=x.device)

    m = mask.unsqueeze(-1).to(x.dtype)           # (N,K,1)
    denom = m.sum(dim=1).clamp_min(1.0)          # (N,1)
    pooled = (x * m).sum(dim=1) / denom          # (N,D)
    return pooled


class MORN(nn.Module):
    """
    Joint-update MORN (no two-stage):
      - Init non-wsi nodes with trainable nn.Embedding(nid)
      - Init wsi nodes from patch features (masked mean pooling + Linear)
      - Run HGT-style MORNLayer on FULL heterograph for n_layers
      - Output head on patient

    Notes:
      - return_patch_attn is kept for API compatibility, but returns None
      - If you really want patch-attn, that reintroduces "two-stage dependency" on patient.
    """
    def __init__(
        self,
        G: dgl.DGLHeteroGraph,
        node_dict: Dict[str, int],
        edge_dict: Dict[Tuple[str, str, str], int],
        n_hid: int,
        n_out: int,
        n_layers: int = 2,
        n_heads: int = 4,
        use_norm: bool = True,
        wsi_patch_dim: int = 1024,
        edge_weight_key: str = "w",
        edge_weight_mode: str = "mul_attn",
        dropout: float = 0.2,
        use_wsi: bool = True,
    ):
        super().__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_hid = n_hid

        self.edge_weight_key = edge_weight_key
        self.edge_weight_mode = edge_weight_mode

        self.use_wsi = bool(use_wsi) and ("wsi" in G.ntypes)

        # trainable embeddings for non-wsi node types
        self.emb = nn.ModuleDict()
        for ntype in G.ntypes:
            if ntype == "wsi":
                continue
            self.emb[ntype] = nn.Embedding(G.num_nodes(ntype), n_hid)

        # wsi init projection (patient-independent)
        if self.use_wsi:
            self.wsi_proj = nn.Linear(wsi_patch_dim, n_hid)

        # one unified stack of message-passing layers on full graph
        self.layers = nn.ModuleList([
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
            for _ in range(int(n_layers))
        ])

        self.out = nn.Linear(n_hid, n_out)

    def _init_h(self, G: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
        """
        Build initial h for all ntypes.
        - non-wsi: embedding(nid)
        - wsi: pooled patches -> proj to n_hid
        """
        h: Dict[str, torch.Tensor] = {}

        # non-wsi types
        for ntype in G.ntypes:
            if ntype == "wsi":
                continue
            nid = G.nodes[ntype].data["nid"]
            h[ntype] = self.emb[ntype](nid)

        # wsi type
        if self.use_wsi:
            patches = G.nodes["wsi"].data["wsi_patches"]               # (N,K,D)
            mask = G.nodes["wsi"].data.get("wsi_patch_mask", None)     # (N,K)
            pooled = masked_mean_pool(patches, mask)                   # (N,D)
            h["wsi"] = F.gelu(self.wsi_proj(pooled))                   # (N,H)

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
        patch_attn = None  # joint-update版本不做patient-query patch attention

        # init hidden states
        h = self._init_h(G)

        # joint message passing on full graph
        for layer in self.layers:
            if return_attn:
                h, attn_dict = layer(G, h, return_attn=True)
                all_edge_attn.append(attn_dict)
            else:
                h = layer(G, h, return_attn=False)

        logits = self.out(h[out_key])

        omics_h = None
        if return_omics_h:
            omics_h = {k: v for k, v in h.items() if k.startswith("gene_")}

        # returns (尽量保持你原来的接口不崩)
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
