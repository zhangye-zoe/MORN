# utils/cluster_constraint.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tensor(x, device=None):
    if torch.is_tensor(x):
        return x.to(device=device) if device is not None else x
    return torch.tensor(x, device=device)


def _parse_cluster_maps(cluster_maps: dict) -> Dict[str, torch.Tensor]:
    """
    Try to parse cluster_maps into:
      cluster_ids[ntype] = LongTensor[num_nodes(ntype)] with cluster_id or -1

    Supported (robust) formats:
    1) cluster_maps["cluster_ids"][ntype] = tensor/list
    2) cluster_maps[ntype] = tensor/list
    3) cluster_maps["ntype_to_cluster_ids"][ntype] = ...
    """
    if cluster_maps is None:
        return {}

    for key in ["cluster_ids", "ntype_to_cluster_ids", "clusters", "cluster_id_by_ntype"]:
        if isinstance(cluster_maps, dict) and key in cluster_maps and isinstance(cluster_maps[key], dict):
            out = {}
            for ntype, arr in cluster_maps[key].items():
                out[str(ntype)] = _as_tensor(arr).long()
            return out

    # fallback: treat top-level keys as ntypes
    out = {}
    for k, v in cluster_maps.items():
        if isinstance(v, (list, tuple)) or torch.is_tensor(v):
            out[str(k)] = _as_tensor(v).long()
    return out


class ClusterSupConLoss(nn.Module):
    """
    Cluster-supervised contrastive loss for node embeddings.

    Goal:
      - nodes in same cluster => close
      - nodes in different clusters => far

    Uses "prototype InfoNCE":
      For each cluster c, compute prototype p_c = mean(z_i in cluster c)
      For each sample z_i, positive = p_{c(i)}, negatives = other prototypes

    This is much more stable & cheaper than all-pairs SupCon.

    Inputs:
      h_dict: dict[ntype] -> Tensor[num_nodes(ntype), hid_dim]
      cluster_maps: dict with cluster id for each node in each ntype
    """
    def __init__(
        self,
        hid_dim: int,
        temperature: float = 0.2,
        proj_dim: int = 128,
        dropout: float = 0.1,
        min_nodes_per_cluster: int = 3,
        max_clusters_per_step: int = 256,
        max_nodes_per_cluster: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.hid_dim = int(hid_dim)
        self.temperature = float(temperature)
        self.min_nodes_per_cluster = int(min_nodes_per_cluster)
        self.max_clusters_per_step = int(max_clusters_per_step)
        self.max_nodes_per_cluster = int(max_nodes_per_cluster)
        self.rng = random.Random(int(seed))

        # projection head helps contrastive stability
        self.proj = nn.Sequential(
            nn.Linear(self.hid_dim, int(proj_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(proj_dim), int(proj_dim)),
        )

    @torch.no_grad()
    def _collect_indices(
        self,
        cluster_ids_by_ntype: Dict[str, torch.Tensor],
        ntypes: List[str],
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
          all_idx: LongTensor[M]  global stacked indices (per-ntype local indices stored separately)
          all_cid: LongTensor[M]  cluster id for each sampled node
        We sample nodes per cluster to bound memory.
        """
        sampled_local = []   # list of (ntype, local_idx_tensor, cid_tensor)

        # gather all clusters across selected ntypes
        cluster_to_members: Dict[int, List[Tuple[str, int]]] = {}

        for ntype in ntypes:
            if ntype not in cluster_ids_by_ntype:
                continue
            cids = cluster_ids_by_ntype[ntype]
            if cids.device != device:
                cids = cids.to(device=device)

            # valid nodes: cid >= 0
            valid = (cids >= 0)
            if valid.sum().item() == 0:
                continue

            idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            cid = cids[idx]

            # build python lists (clusters are usually not too many)
            for i, c in zip(idx.tolist(), cid.tolist()):
                cluster_to_members.setdefault(int(c), []).append((ntype, int(i)))

        # filter by min size
        clusters = [c for c, mem in cluster_to_members.items() if len(mem) >= self.min_nodes_per_cluster]
        if len(clusters) == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

        # sample clusters per step
        if len(clusters) > self.max_clusters_per_step:
            clusters = self.rng.sample(clusters, self.max_clusters_per_step)

        # sample nodes within each cluster
        for c in clusters:
            members = cluster_to_members[c]
            if len(members) > self.max_nodes_per_cluster:
                members = self.rng.sample(members, self.max_nodes_per_cluster)

            # group by ntype for efficiency
            by_ntype: Dict[str, List[int]] = {}
            for ntype, i in members:
                by_ntype.setdefault(ntype, []).append(i)

            for ntype, idx_list in by_ntype.items():
                local_idx = torch.tensor(idx_list, dtype=torch.long, device=device)
                cid_tensor = torch.full((local_idx.numel(),), int(c), dtype=torch.long, device=device)
                sampled_local.append((ntype, local_idx, cid_tensor))

        # return packed lists (we keep (ntype, idx) structure outside)
        # We'll just return as two lists and rebuild in forward.
        # Here encode ntype as string list stored in python in forward.
        # (No need for global idx)
        # We return empty placeholders, real work in forward.
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

    def forward(
        self,
        h_dict: Dict[str, torch.Tensor],
        cluster_maps: dict,
        ntypes: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute cluster prototype InfoNCE loss.
        """
        if cluster_maps is None:
            return torch.tensor(0.0, device=next(iter(h_dict.values())).device)

        device = next(iter(h_dict.values())).device
        cluster_ids_by_ntype = _parse_cluster_maps(cluster_maps)
        if len(cluster_ids_by_ntype) == 0:
            return torch.tensor(0.0, device=device)

        if ntypes is None:
            # default: all gene_* types present in h_dict
            ntypes = [k for k in h_dict.keys() if k.startswith("gene_")]
        else:
            ntypes = [x for x in ntypes if x in h_dict]

        if len(ntypes) == 0:
            return torch.tensor(0.0, device=device)

        # Build sampled node embeddings grouped by cluster id
        cluster_to_z: Dict[int, List[torch.Tensor]] = {}
        cluster_to_count: Dict[int, int] = {}

        # We do deterministic-ish sampling on GPU-friendly ops:
        # For each ntype: gather indices per cluster; then sample (cap) by random permutation.
        for ntype in ntypes:
            if ntype not in cluster_ids_by_ntype:
                continue
            if ntype not in h_dict:
                continue

            H = h_dict[ntype]  # (N,D)
            cids = cluster_ids_by_ntype[ntype].to(device=device).long()

            N = H.shape[0]
            if cids.numel() != N:
                # mismatch => skip (avoid silent wrong indexing)
                continue

            valid = (cids >= 0)
            if valid.sum().item() == 0:
                continue

            idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            cid = cids[idx]

            # random permute to sample limited nodes per cluster
            perm = torch.randperm(idx.numel(), device=device)
            idx = idx[perm]
            cid = cid[perm]

            # collect to python dict (clusters are usually manageable)
            for i, c in zip(idx.tolist(), cid.tolist()):
                c = int(c)
                cluster_to_count[c] = cluster_to_count.get(c, 0) + 1
                if cluster_to_count[c] <= self.max_nodes_per_cluster:
                    cluster_to_z.setdefault(c, []).append(H[i])

        # filter by min size
        clusters = [c for c, zs in cluster_to_z.items() if len(zs) >= self.min_nodes_per_cluster]
        if len(clusters) == 0:
            return torch.tensor(0.0, device=device)

        # sample clusters per step
        if len(clusters) > self.max_clusters_per_step:
            clusters = self.rng.sample(clusters, self.max_clusters_per_step)

        # Stack samples
        z_list = []
        y_list = []
        for c in clusters:
            zs = cluster_to_z[c]
            for z in zs:
                z_list.append(z)
                y_list.append(c)

        z = torch.stack(z_list, dim=0)  # (M,D)
        y = torch.tensor(y_list, dtype=torch.long, device=device)  # (M,)

        # project & normalize
        z = self.proj(z)
        z = F.normalize(z, dim=1)

        # prototypes
        uniq = torch.unique(y)
        # build prototype per cluster
        protos = []
        proto_cids = []
        for c in uniq.tolist():
            m = (y == c)
            p = z[m].mean(dim=0)
            protos.append(p)
            proto_cids.append(int(c))
        P = torch.stack(protos, dim=0)            # (C,proj_dim)
        P = F.normalize(P, dim=1)
        proto_cids = torch.tensor(proto_cids, dtype=torch.long, device=device)  # (C,)

        # logits: (M,C)
        logits = (z @ P.t()) / self.temperature

        # target index per sample
        # map cluster id -> prototype row
        cid_to_row = {int(c.item()): i for i, c in enumerate(proto_cids)}
        target = torch.tensor([cid_to_row[int(c.item())] for c in y], dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, target)
        return loss
