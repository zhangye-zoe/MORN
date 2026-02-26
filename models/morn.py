# models/morn.py
import math
from typing import Dict, Tuple, Optional, List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.morn_layer import MORNLayer
from models.cross_attention import MMAttentionLayer, FeedForward


CanonicalEType = Tuple[str, str, str]


class PatientQueryPatchAttn(nn.Module):
    """
    patient query -> attend over patches to get wsi token
    patches: (N, K, D_in)
    mask:    (N, K) bool
    query_h: (N, H)
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
        p = F.gelu(self.patch_proj(patches))      # (N,K,H)
        q = self.q_proj(query_h).unsqueeze(1)     # (N,1,H)
        k = self.k_proj(p)                        # (N,K,H)
        v = self.v_proj(p)                        # (N,K,H)

        # print('p', p)
        # print('q', q)
        # print('k', k)
        # print(q.shape, k.shape)
        # print('=' * 100)
        # print('scaling factor', self.scale)
        # print('mmm', (q * k).sum(dim=-1).squeeze(1), (q * k).sum(dim=-1).squeeze(1).shape)  # (N,K)

        # def check_cpu(x, name):
        #     if x is None:
        #         print(f"[{name}] is None")
        #         return
        #     if not torch.is_tensor(x):
        #         print(f"[{name}] type={type(x)}")
        #         return
        #     print(f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
        #     # 关键：搬到 CPU 再做 finite 检查
        #     xc = x.detach().float().cpu()
        #     if xc.numel() == 0:
        #         print(f"[{name}] EMPTY")
        #         return
        #     isfin = torch.isfinite(xc)
        #     if not isfin.all():
        #         bad = (~isfin).nonzero(as_tuple=False)[:5]
        #         i = bad[0].tolist()
        #         print(f"[{name}] NaN/Inf at {i}, value={xc[tuple(i)].item()}")
        #         raise RuntimeError(f"{name} has NaN/Inf")
        #     print(f"[{name}] finite OK, min={xc.min().item():.4g}, max={xc.max().item():.4g}")

        # check_cpu(q,"q"); check_cpu(k,"k")

        # score = (q * k).sum(dim=-1).squeeze(1) / self.scale  # (N,K)
        # q: (N,1,D), k: (N,K,D)
        score = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / self.scale  # (N,K)
        # print('score', score)
        # print('=' * 100)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            mask = mask.to(score.device)
            score = score.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(score, dim=1)        # (N,K)
        attn = self.drop(attn)
        wsi_emb = torch.bmm(attn.unsqueeze(1), v).squeeze(1)  # (N,H)
        return wsi_emb, attn


class MORN(nn.Module):
    """
    ✅ Key constraints:
    - NO patient embedding table (patient is not randomly initialized)
    - patient never acts as message SOURCE (no patient->gene propagation)
    - stage-wise: omics warmup then enable WSI mixer
    - SurvPath-style token mixer: omics tokens + wsi token
    """
    def __init__(
        self,
        G: dgl.DGLHeteroGraph,
        node_dict: Dict[str, int],
        edge_dict: Dict[CanonicalEType, int],
        n_hid: int,
        n_out: int,
        n_layers_omics: int = 2,
        n_layers_cross: int = 0,  # v2 default: do NOT do graph-cross; we do token mixer instead
        n_heads: int = 4,
        use_norm: bool = True,
        wsi_patch_dim: int = 1024,
        dropout: float = 0.2,
        edge_weight_key: str = "w",
        edge_weight_mode: str = "mul_attn",
        use_wsi: bool = True,
        omics_token_ntypes: Optional[List[str]] = None,  # e.g., ["gene_CNV","gene_Methy","gene_miRNA","gene_mRNA"]
    ):
        super().__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_hid = n_hid
        self.n_out = n_out
        self.edge_weight_key = edge_weight_key
        self.edge_weight_mode = edge_weight_mode

        # wsi availability
        self.use_wsi = bool(use_wsi) and ("wsi" in G.ntypes)

        # ✅ learnable embeddings for NON-patient nodes (gene_*, pathway, etc.)
        # patient has NO embedding; patient h starts from zeros and only receives messages
        self.emb = nn.ModuleDict()
        for ntype in G.ntypes:
            if ntype == "patient":
                continue
            if ntype == "wsi":
                continue
            self.emb[ntype] = nn.Embedding(G.num_nodes(ntype), n_hid)

        # omics encoder (message passing)
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

        # optional cross graph layers (not recommended in v2, token mixer preferred)
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

        # WSI patch attention
        self.patch_attn = PatientQueryPatchAttn(
            patch_dim=wsi_patch_dim,
            hid_dim=n_hid,
            dropout=dropout,
        )

        # which omics ntypes become "tokens" for SurvPath-style mixer
        if omics_token_ntypes is None:
            # auto pick by prefix
            omics_token_ntypes = [nt for nt in G.ntypes if nt.startswith("gene_")]
            omics_token_ntypes = sorted(omics_token_ntypes)
        self.omics_token_ntypes = omics_token_ntypes

        # token mixer (SurvPath MMAttentionLayer):
        # tokens = [omics_token_1 ... omics_token_M, wsi_token]
        self.num_pathways = len(self.omics_token_ntypes)
        self.token_mixer = MMAttentionLayer(
            dim=n_hid,
            dim_head=max(1, n_hid // 2),
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_pathways,
        )
        self.ff = FeedForward(n_hid // 2, dropout=dropout)
        self.ln = nn.LayerNorm(n_hid // 2)

        # head (SurvPath-style): concat(mean(omics tokens), mean(wsi tokens)) -> logits
        self.to_logits = nn.Sequential(
            nn.Linear(n_hid, n_hid // 4),
            nn.ReLU(),
            nn.Linear(n_hid // 4, n_out),
        )

        # phase flag
        self._phase = "omics"  # or "full"

    def set_phase(self, phase: str):
        assert phase in ["omics", "full"]
        self._phase = phase

    def _init_h(self, G: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
        """
        patient: zeros (no embedding)
        others: embedding(nid)
        """
        # infer device/dtype
        dev = G.device
        h: Dict[str, torch.Tensor] = {}

        for ntype in G.ntypes:
            if ntype == "patient":
                h[ntype] = torch.zeros(G.num_nodes(ntype), self.n_hid, device=dev)
            elif ntype == "wsi":
                # wsi node embedding is produced later (from patches) when needed
                continue
            else:
                nid = G.nodes[ntype].data["nid"].to(dev)
                h[ntype] = self.emb[ntype](nid)
        return h

    def _filter_etypes_no_patient_src(self, G: dgl.DGLHeteroGraph) -> List[CanonicalEType]:
        """
        ✅ forbid any relation whose srctype == patient
        This is the key to prevent patient->gene leakage.
        """
        keep = []
        for et in G.canonical_etypes:
            s, _, _ = et
            if s == "patient":
                continue
            keep.append(et)
        return keep

    def encode_omics(
        self,
        G: dgl.DGLHeteroGraph,
        return_attn: bool = False,
    ):
        """
        Omics message passing with patient as receiver only.
        """
        all_attn = [] if return_attn else None
        h = self._init_h(G)

        # build subgraph excluding patient-as-source etypes
        keep_etypes = self._filter_etypes_no_patient_src(G)
        if len(keep_etypes) == 0 or len(self.layers_omics) == 0:
            return h, all_attn

        G_mp = dgl.edge_type_subgraph(G, keep_etypes)
        for layer in self.layers_omics:
            if return_attn:
                h, attn = layer(G_mp, h, return_attn=True)
                all_attn.append(attn)
            else:
                h = layer(G_mp, h, return_attn=False)

        return h, all_attn

    def build_tokens_from_h(self, h: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build omics tokens (N, M, H) by pooling each gene_omic type embedding into one token.
        A simple but stable design: token_i = mean over all nodes of that omics ntype.
        If you have pathway mapping, you can replace this with pathway-level pooling later.
        """
        tokens = []
        for nt in self.omics_token_ntypes:
            if nt not in h:
                # missing under ablation
                continue
            tok = h[nt].mean(dim=0, keepdim=True)  # (1,H)
            tokens.append(tok)
        if len(tokens) == 0:
            # fallback: if no gene_ types, use patient embedding as 1 token
            tokens = [h["patient"].mean(dim=0, keepdim=True)]
        toks = torch.cat(tokens, dim=0)  # (M,H)
        # broadcast to each patient as same global tokens; you can switch to patient-specific tokens later
        Np = h["patient"].size(0)
        return toks.unsqueeze(0).expand(Np, -1, -1).contiguous()  # (Np,M,H)

    def forward(
        self,
        G: dgl.DGLHeteroGraph,
        target_ntype: str = "patient",
        return_attn: bool = False,
        return_patch_attn: bool = False,
        return_omics_h: bool = False,
        return_patient_embed: bool = False,  # omics patient embedding for CL
    ):
        """
        Returns:
          logits: (N_patient, n_out)
          optionally: all_attn, patch_attn, omics_h, patient_embed
        """
        patch_attn = None
        all_attn_total = [] if return_attn else None

        # --- Stage A: omics encoder ---
        h, all_attn_omics = self.encode_omics(G, return_attn=return_attn)
        # print('h', h)
        # print('=' * 100)
        if return_attn:
            all_attn_total += all_attn_omics

        omics_h = None
        if return_omics_h:
            omics_h = {k: v for k, v in h.items() if k.startswith("gene_")}

        patient_embed = h[target_ntype]  # patient embedding after omics aggregation

        # --- Phase decision ---
        phase = self._phase

        # --- Stage B: WSI mixer (optional, only in full phase) ---
        if (phase == "full") and self.use_wsi:
            # patient-query patch attention -> wsi token
            patches = G.nodes["wsi"].data["wsi_patches"]
            mask = G.nodes["wsi"].data.get("wsi_patch_mask", None)
            # print('patches', patches)
            # print('mask', mask)
            # print('=' * 100)
            wsi_tok, patch_attn = self.patch_attn(patches, mask, patient_embed)  # (Np,H), (Np,K)


            # build omics tokens (Np, M, H)
            omics_tokens = self.build_tokens_from_h(h)  # (Np,M,H)
            # print('omics_tokens', omics_tokens)
            # print('=' * 20)
            # print('wsi_tok', wsi_tok)
            # print('=' * 100)

            # tokens = [omics_tokens, wsi_token_as_1_token]
            tokens = torch.cat([omics_tokens, wsi_tok.unsqueeze(1)], dim=1)  # (Np, M+1, H)

            # SurvPath-style mixer
            mm = self.token_mixer(x=tokens, mask=None, return_attention=False)  # (Np,M+1,H)
            mm = self.ff(mm)
            mm = self.ln(mm)

            # aggregate like SurvPath
            M = mm.size(1) - 1
            om = mm[:, :M, :].mean(dim=1)         # (Np,H')
            ws = mm[:, M:, :].mean(dim=1)         # (Np,H')
            fused = torch.cat([om, ws], dim=1)    # (Np, 2*H') but our to_logits expects H -> keep H
            # We keep dims aligned by ensuring mixer outputs H, not H//2 in your FeedForward;
            # if your FeedForward is H//2, you can replace ff/ln with Identity.
            # For safety: project back to n_hid
            fused = F.pad(fused, (0, self.n_hid - fused.size(1))) if fused.size(1) < self.n_hid else fused[:, :self.n_hid]

            logits = self.to_logits(fused)        # (Np,n_out)
        else:
            # omics-only prediction (stable): directly from patient_embed
            logits = self.to_logits(patient_embed)

        # returns
        outs = [logits]
        if return_attn:
            outs.append(all_attn_total)
        if return_patch_attn:
            outs.append(patch_attn)
        if return_omics_h:
            outs.append(omics_h)
        if return_patient_embed:
            outs.append(patient_embed)

        return outs[0] if len(outs) == 1 else tuple(outs)
