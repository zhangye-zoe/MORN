#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import dgl


# -------------------------
# small helpers
# -------------------------
def _get_device_of_module(m: nn.Module) -> torch.device:
    for p in m.parameters():
        return p.device
    return torch.device("cpu")


def _safe_edge_weight(g: dgl.DGLHeteroGraph, etype: Tuple[str, str, str], w_key: str) -> Optional[torch.Tensor]:
    if w_key is None:
        return None
    if w_key in g.edges[etype].data:
        w = g.edges[etype].data[w_key]
        if not torch.is_floating_point(w):
            w = w.float()
        return w
    return torch.ones(g.num_edges(etype), device=g.device, dtype=torch.float32)


def _ensure_nid_tensor(G: dgl.DGLHeteroGraph, ntype: str, target_device: torch.device) -> torch.Tensor:
    if "nid" in G.nodes[ntype].data:
        nid = G.nodes[ntype].data["nid"]
    elif dgl.NID in G.nodes[ntype].data:
        nid = G.nodes[ntype].data[dgl.NID]
    else:
        nid = torch.arange(G.num_nodes(ntype), dtype=torch.long, device=G.device)
    return nid.long().to(target_device, non_blocking=True)


# -------------------------
# Regulation encoder (simple, stable)
# -------------------------
class SimpleRegEncoder(nn.Module):
    """
    Weighted message passing on enabled regulation edge types.
    IMPORTANT: all graph features must be on G.device.
    """

    def __init__(self, hid_dim: int, dropout: float = 0.2):
        super().__init__()
        self.hid_dim = int(hid_dim)
        self.dropout = float(dropout)
        self.lin = nn.ModuleDict()
        self.act = nn.GELU()
        self.do = nn.Dropout(self.dropout)

    def reset_node_types(self, ntypes: List[str]):
        for nt in ntypes:
            if nt not in self.lin:
                self.lin[nt] = nn.Linear(self.hid_dim, self.hid_dim)

    def forward(
        self,
        G: dgl.DGLHeteroGraph,
        h: Dict[str, torch.Tensor],
        enabled_reg_etypes: List[Tuple[str, str, str]],
        w_key: str = "w",
    ) -> Dict[str, torch.Tensor]:
        if len(enabled_reg_etypes) == 0:
            return h

        gdev = G.device  # ✅ graph device (cpu in your env)
        self.reset_node_types([et[0] for et in enabled_reg_etypes] + [et[2] for et in enabled_reg_etypes])

        with G.local_scope():
            # attach features on graph device
            for nt, feat in h.items():
                if nt in G.ntypes:
                    G.nodes[nt].data["h"] = feat.to(gdev, non_blocking=True)

            funcs = {}
            for et in enabled_reg_etypes:
                if et not in G.canonical_etypes:
                    continue
                src, _, dst = et
                if src not in h or dst not in h:
                    continue

                w = _safe_edge_weight(G, et, w_key)
                if w is not None:
                    w = w.to(gdev, non_blocking=True)

                def _msg(edges, _w=w):
                    m = edges.src["h"]
                    if _w is not None:
                        m = m * _w.unsqueeze(-1)
                    return {"m": m}

                funcs[et] = (_msg, dgl.function.sum("m", "agg"))

            if len(funcs) == 0:
                return h

            G.multi_update_all(funcs, cross_reducer="sum")

            out = dict(h)
            for et in enabled_reg_etypes:
                if et not in funcs:
                    continue
                dst = et[2]
                if "agg" not in G.nodes[dst].data:
                    continue
                agg = G.nodes[dst].data["agg"]  # on gdev
                base = out[dst].to(gdev, non_blocking=True)
                upd = self.lin[dst](agg)
                upd = self.act(upd)
                upd = self.do(upd)
                out[dst] = base + upd

            return out


# -------------------------
# Patient readout (gene -> patient aggregation)
# -------------------------
class PatientReadout(nn.Module):
    """
    IMPORTANT: all graph features must be on G.device.
    Output will be on G.device.
    """

    def __init__(self, hid_dim: int, dropout: float = 0.2):
        super().__init__()
        self.hid_dim = int(hid_dim)
        self.dropout = float(dropout)

        self.proj = nn.Linear(self.hid_dim, self.hid_dim)
        self.norm = nn.LayerNorm(self.hid_dim)
        self.act = nn.GELU()
        self.do = nn.Dropout(self.dropout)

    def forward(
        self,
        G: dgl.DGLHeteroGraph,
        gene_h: Dict[str, torch.Tensor],
        patient_ntype: str = "patient",
        w_key: str = "w",
    ) -> torch.Tensor:
        assert patient_ntype in G.ntypes
        gdev = G.device  # ✅ graph device

        with G.local_scope():
            # attach gene h on graph device
            for nt, feat in gene_h.items():
                if nt in G.ntypes:
                    G.nodes[nt].data["h"] = feat.to(gdev, non_blocking=True)

            # init patient accumulator on graph device
            G.nodes[patient_ntype].data["acc"] = torch.zeros(
                (G.num_nodes(patient_ntype), self.hid_dim),
                device=gdev,
                dtype=torch.float32,
            )

            funcs = {}
            for et in G.canonical_etypes:
                src, _, dst = et
                if dst != patient_ntype:
                    continue
                if src not in gene_h:
                    continue

                w = _safe_edge_weight(G, et, w_key)
                if w is not None:
                    w = w.to(gdev, non_blocking=True)

                def _msg(edges, _w=w):
                    m = edges.src["h"]
                    if _w is not None:
                        m = m * _w.unsqueeze(-1)
                    return {"m": m}

                funcs[et] = (_msg, dgl.function.sum("m", "acc_part"))

            if len(funcs) > 0:
                G.multi_update_all(funcs, cross_reducer="sum")
                if "acc_part" in G.nodes[patient_ntype].data:
                    G.nodes[patient_ntype].data["acc"] = G.nodes[patient_ntype].data["acc_part"]

            pat = G.nodes[patient_ntype].data["acc"]  # on gdev
            pat = self.proj(pat)
            pat = self.act(pat)
            pat = self.do(pat)
            pat = self.norm(pat)
            return pat


# -------------------------
# WSI encoder
# -------------------------
class SimpleWSIEncoder(nn.Module):
    def __init__(self, patch_dim: int, hid_dim: int, dropout: float = 0.2):
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.hid_dim = int(hid_dim)
        self.dropout = float(dropout)

        self.lin = nn.Linear(self.patch_dim, self.hid_dim)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.norm = nn.LayerNorm(self.hid_dim)

    def forward(self, patches: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.lin(patches)
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            m = mask.to(x.device).float()
            denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (x * m.unsqueeze(-1)).sum(dim=1) / denom
        pooled = self.mlp(pooled)
        pooled = self.norm(pooled)
        return pooled


# -------------------------
# Main model
# -------------------------
class MORN_SurvPathStyle(nn.Module):
    """
    Dual-device safe:
      - Omics branch (emb/reg/readout/omics_head) follows GRAPH device (likely CPU in your DGL build).
      - WSI + fusion runs on fusion_device (usually CUDA).
    """

    def __init__(
        self,
        hid_dim: int,
        n_bins: int,
        use_omics: Dict[str, bool],
        use_regulation: Dict[str, bool],
        wsi_patch_dim: int,
        dropout: float = 0.2,
        detach_omics_query: bool = True,
    ):
        super().__init__()
        self.hid_dim = int(hid_dim)
        self.n_bins = int(n_bins)
        self.use_omics = dict(use_omics)
        self.use_regulation = dict(use_regulation)
        self.wsi_patch_dim = int(wsi_patch_dim)
        self.dropout = float(dropout)
        self.detach_omics_query = bool(detach_omics_query)

        self.emb = nn.ModuleDict()

        self.reg_encoder = SimpleRegEncoder(self.hid_dim, dropout=self.dropout)
        self.readout = PatientReadout(self.hid_dim, dropout=self.dropout)
        self.omics_head = nn.Linear(self.hid_dim, self.n_bins)

        self.wsi_encoder = SimpleWSIEncoder(self.wsi_patch_dim, self.hid_dim, dropout=self.dropout)
        self.fuse_head = nn.Sequential(
            nn.Linear(self.hid_dim * 2, self.hid_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid_dim, self.n_bins),
        )

        # devices (can be set by set_devices)
        self.graph_device = torch.device("cpu")
        self.fusion_device = _get_device_of_module(self)

    def set_devices(self, graph_device: torch.device, fusion_device: torch.device):
        """
        Call once BEFORE creating optimizer.
        - graph_device: where DGL graph lives (cpu in your env)
        - fusion_device: where WSI+fusion runs (cuda)
        """
        self.graph_device = torch.device(graph_device)
        self.fusion_device = torch.device(fusion_device)

        # move omics branch to graph_device
        for k, m in self.emb.items():
            self.emb[k] = m.to(self.graph_device)
        self.reg_encoder.to(self.graph_device)
        self.readout.to(self.graph_device)
        self.omics_head.to(self.graph_device)

        # move wsi+fusion to fusion_device
        self.wsi_encoder.to(self.fusion_device)
        self.fuse_head.to(self.fusion_device)

    def reset_num_embeddings(self, num_nodes_dict: Dict[str, int]):
        for nt, n in num_nodes_dict.items():
            n = int(n)
            if n <= 0:
                continue
            if nt not in self.emb or self.emb[nt].num_embeddings != n:
                self.emb[nt] = nn.Embedding(n, self.hid_dim)
                nn.init.normal_(self.emb[nt].weight, std=0.02)

        # keep embeddings on graph_device (important)
        for nt in list(self.emb.keys()):
            self.emb[nt] = self.emb[nt].to(self.graph_device)

    def _init_gene_h(self, G: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
        gdev = G.device  # must match embeddings device (graph_device)
        gene_h: Dict[str, torch.Tensor] = {}
        for nt in G.ntypes:
            if nt not in self.emb:
                continue
            if nt.startswith("gene_") and (not self.use_omics.get(nt, True)):
                continue
            emb_layer = self.emb[nt]
            nid = _ensure_nid_tensor(G, nt, emb_layer.weight.device)  # should be gdev
            gene_h[nt] = emb_layer(nid).to(gdev, non_blocking=True)
        return gene_h

    def _enabled_reg_etypes(self, G: dgl.DGLHeteroGraph) -> List[Tuple[str, str, str]]:
        enabled = []
        if self.use_regulation.get("mti", False):
            for et in [
                ("gene_miRNA", "targets", "gene_mRNA"),
                ("gene_mRNA", "targeted_by", "gene_miRNA"),
            ]:
                if et in G.canonical_etypes:
                    enabled.append(et)
        if self.use_regulation.get("pathway_hub", False):
            for et in [
                ("gene_CNV", "cnv_to_mrna", "gene_mRNA"),
                ("gene_mRNA", "mrna_to_cnv", "gene_CNV"),
                ("gene_Methy", "methy_to_mrna", "gene_mRNA"),
                ("gene_mRNA", "mrna_to_methy", "gene_Methy"),
            ]:
                if et in G.canonical_etypes:
                    enabled.append(et)
        return enabled

    def forward_omics_only(self, G_batch: dgl.DGLHeteroGraph, w_key: str = "w"):
        # ✅ 不强行把图搬 GPU；omics 跟随图设备
        gene_h = self._init_gene_h(G_batch)

        reg_etypes = self._enabled_reg_etypes(G_batch)
        if len(reg_etypes) > 0:
            gene_h = self.reg_encoder(G_batch, gene_h, reg_etypes, w_key=w_key)

        patient_omics = self.readout(G_batch, gene_h, patient_ntype="patient", w_key=w_key)
        logits = self.omics_head(patient_omics)
        return patient_omics, logits, gene_h

    def forward_with_wsi(
        self,
        G_batch: dgl.DGLHeteroGraph,
        wsi_patches: torch.Tensor,
        wsi_mask: Optional[torch.Tensor] = None,
        w_key: str = "w",
    ):
        # omics on graph device (likely cpu)
        patient_omics, _logits_omics, gene_h = self.forward_omics_only(G_batch, w_key=w_key)

        # wsi on fusion device (likely cuda)
        wsi_patches = wsi_patches.to(self.fusion_device, non_blocking=True)
        if wsi_mask is not None:
            wsi_mask = wsi_mask.to(self.fusion_device, non_blocking=True)
        wsi_emb = self.wsi_encoder(wsi_patches, wsi_mask)

        # move omics embedding to fusion device for concat
        patient_omics_fuse = patient_omics.detach() if self.detach_omics_query else patient_omics
        patient_omics_fuse = patient_omics_fuse.to(self.fusion_device, non_blocking=True)

        fused = torch.cat([patient_omics_fuse, wsi_emb], dim=1)
        logits = self.fuse_head(fused)

        attn = None
        return logits, patient_omics_fuse, wsi_emb, attn, gene_h
