# models/morn_patient.py
import math
from typing import Dict, Tuple, Optional, List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.morn_layer_patient import MORNLayer


def masked_mean_pool(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    if mask.dtype != torch.bool:
        mask = mask.bool()
    mask = mask.to(x.device)
    m = mask.unsqueeze(-1).to(x.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom


class MORNPatient(nn.Module):
    """
    Per-patient graph model (NO patient-patient leakage):
      - Graph contains exactly 1 patient node (local)
      - Omics nodes are only those linked to this patient
      - Optional WSI node is only this patient's WSI
      - Regulation edges are intra-patient only

    Initialization:
      - Each ntype gets a trainable "type token" vector, broadcast to its nodes
      - Optional: wsi node uses pooled patch feature -> linear -> added to type token
    """
    def __init__(
        self,
        ntypes: List[str],
        canonical_etypes: List[Tuple[str, str, str]],
        node_dict: Dict[str, int],
        edge_dict: Dict[Tuple[str, str, str], int],
        n_hid: int,
        n_out: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        use_norm: bool = True,
        use_wsi: bool = True,
        wsi_patch_dim: int = 1024,
        edge_weight_key: str = "w",
        edge_weight_mode: str = "dst_norm",
    ):
        super().__init__()
        self.ntypes = list(ntypes)
        self.canonical_etypes = list(canonical_etypes)
        self.node_dict = dict(node_dict)
        self.edge_dict = dict(edge_dict)

        self.n_hid = int(n_hid)
        self.n_out = int(n_out)
        self.use_wsi = bool(use_wsi) and ("wsi" in self.ntypes)

        # type token embeddings (1 per ntype)
        self.type_token = nn.ParameterDict({
            nt: nn.Parameter(torch.zeros(self.n_hid)) for nt in self.ntypes
        })
        for nt in self.ntypes:
            nn.init.normal_(self.type_token[nt], mean=0.0, std=0.02)

        # optional wsi feature projection
        if self.use_wsi:
            self.wsi_proj = nn.Linear(int(wsi_patch_dim), self.n_hid)

        self.layers = nn.ModuleList([
            MORNLayer(
                in_dim=self.n_hid,
                out_dim=self.n_hid,
                node_dict=self.node_dict,
                edge_dict=self.edge_dict,
                n_heads=int(n_heads),
                dropout=float(dropout),
                use_norm=bool(use_norm),
                edge_weight_key=edge_weight_key,
                edge_weight_mode=edge_weight_mode,
            )
            for _ in range(int(n_layers))
        ])

        self.out = nn.Linear(self.n_hid, self.n_out)

    def _init_h(self, G: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
        h = {}
        for nt in G.ntypes:
            tok = self.type_token[nt].unsqueeze(0)  # (1,H)
            h[nt] = tok.repeat(G.num_nodes(nt), 1)

        # add wsi pooled feature if enabled
        if self.use_wsi and ("wsi" in G.ntypes) and ("wsi_patches" in G.nodes["wsi"].data):
            patches = G.nodes["wsi"].data["wsi_patches"]          # (N,K,D)
            mask = G.nodes["wsi"].data.get("wsi_patch_mask", None)
            pooled = masked_mean_pool(patches, mask)              # (N,D)
            h["wsi"] = h["wsi"] + F.gelu(self.wsi_proj(pooled))   # add feature to token

        return h

    def forward(self, G: dgl.DGLHeteroGraph, out_key: str = "patient", return_attn: bool = False):
        all_attn = [] if return_attn else None
        h = self._init_h(G)

        for layer in self.layers:
            if return_attn:
                h, attn_dict = layer(G, h, return_attn=True)
                all_attn.append(attn_dict)
            else:
                h = layer(G, h, return_attn=False)

        logits = self.out(h[out_key])
        if return_attn:
            return logits, all_attn
        return logits
