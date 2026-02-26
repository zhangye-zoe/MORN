#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class OmicsProjHead(nn.Module):
    """Small projection head for contrastive learning."""
    def __init__(self, in_dim: int, proj_dim: int = 128, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PathwaySupConLoss(nn.Module):
    """
    Pathway-guided multi-view supervised contrastive learning.

    Given gene embeddings for multiple omics types, we build pathway prototypes by mean pooling
    gene embeddings belonging to each pathway, and enforce that the SAME pathway across different omics
    are close (positives), while different pathways are far (negatives).

    This is a "SupCon" style loss:
      - anchor: each (pathway p, omics o) prototype
      - positives: same pathway p, other omics o'
      - negatives: different pathway q != p, any omics
    """
    def __init__(
        self,
        hid_dim: int,
        omics_ntypes: List[str],
        proj_dim: int = 128,
        temperature: float = 0.1,
        min_genes_per_pathway: int = 5,
        pathways_per_step: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.omics_ntypes = list(omics_ntypes)
        self.temperature = float(temperature)
        self.min_genes_per_pathway = int(min_genes_per_pathway)
        self.pathways_per_step = int(pathways_per_step)

        # per-omics projection heads (more stable than sharing one)
        self.proj = nn.ModuleDict({
            ntype: OmicsProjHead(hid_dim, proj_dim=proj_dim, hidden_dim=hid_dim, dropout=dropout)
            for ntype in self.omics_ntypes
        })

    @staticmethod
    def _mean_pool(h: torch.Tensor, idx: torch.LongTensor) -> Optional[torch.Tensor]:
        if idx.numel() == 0:
            return None
        return h.index_select(0, idx).mean(dim=0)

    def _sample_pathways(self, valid_pairs: List[Tuple[int, List[str]]], k: int) -> List[Tuple[int, List[str]]]:
        if len(valid_pairs) == 0:
            return []
        if len(valid_pairs) <= k:
            return valid_pairs
        return random.sample(valid_pairs, k)

    def forward(
        self,
        omics_h: Dict[str, torch.Tensor],
        pathway_maps: Dict,
        device: torch.device,
    ) -> torch.Tensor:
        """
        omics_h: dict {ntype: (N_genes, hid_dim)} from Stage-A omics-only graph
        pathway_maps: loaded from pathway_maps.pt, containing:
            - pathway_names: List[str]
            - maps: Dict[str, List[LongTensor]]  maps[ntype][p] = gene_idx tensor
        """
        maps = pathway_maps["maps"]  # Dict[str, List[LongTensor]]
        pathway_names = pathway_maps["pathway_names"]
        P = len(pathway_names)

        # Build list of (pathway_id, available_omics_ntypes>=2)
        valid_pairs: List[Tuple[int, List[str]]] = []
        for p in range(P):
            avail = []
            for ntype in self.omics_ntypes:
                if ntype not in maps:
                    continue
                idx = maps[ntype][p]
                if idx is None:
                    continue
                if idx.numel() >= self.min_genes_per_pathway:
                    # also need that omics_h has that ntype
                    if ntype in omics_h and omics_h[ntype] is not None:
                        avail.append(ntype)
            if len(avail) >= 2:
                valid_pairs.append((p, avail))

        sampled = self._sample_pathways(valid_pairs, self.pathways_per_step)
        if len(sampled) == 0:
            # no usable pathways -> no CL signal
            return torch.zeros((), device=device, dtype=torch.float32)

        # Build prototypes for each (p, ntype)
        feats = []
        labels = []  # pathway id for SupCon positives
        for p, avail in sampled:
            for ntype in avail:
                idx = maps[ntype][p].to(device=device)
                proto = self._mean_pool(omics_h[ntype], idx)
                if proto is None:
                    continue
                z = self.proj[ntype](proto.unsqueeze(0)).squeeze(0)  # (proj_dim,)
                feats.append(z)
                labels.append(p)

        if len(feats) < 4:
            # too few prototypes to form stable contrastive batches
            return torch.zeros((), device=device, dtype=torch.float32)

        z = torch.stack(feats, dim=0)  # (M, proj_dim)
        z = _l2norm(z)
        y = torch.tensor(labels, device=device, dtype=torch.long)  # (M,)

        # cosine similarity / temperature
        sim = (z @ z.t()) / self.temperature  # (M,M)

        # mask self
        M = sim.size(0)
        self_mask = torch.eye(M, device=device, dtype=torch.bool)
        sim = sim.masked_fill(self_mask, float("-inf"))

        # positive mask: same pathway id
        pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & (~self_mask)  # (M,M)

        # SupCon loss:
        # for each i: - log( sum_{pos} exp(sim_ij) / sum_{all != i} exp(sim_ik) )
        exp_sim = torch.exp(sim)  # exp(-inf)=0
        denom = exp_sim.sum(dim=1) + 1e-12  # (M,)

        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1) + 1e-12  # (M,)
        # only count anchors that have at least one positive
        valid_anchor = pos_mask.sum(dim=1) > 0
        loss = -torch.log(pos_sum[valid_anchor] / denom[valid_anchor])
        return loss.mean()
