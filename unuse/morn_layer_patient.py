# models/morn_layer_patient.py
import math
from typing import Dict, Tuple, Optional

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax


class MORNLayer(nn.Module):
    """
    HGT-style hetero attention layer (ablation-safe).
    Supports edge weight normalization by dst (recommended for expression-derived weights).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        node_dict: Dict[str, int],
        edge_dict: Dict[Tuple[str, str, str], int],
        n_heads: int,
        dropout: float = 0.2,
        use_norm: bool = True,
        edge_weight_key: str = "w",
        edge_weight_mode: str = "dst_norm",  # "dst_norm" | "mul_attn" | "none"
        eps: float = 1e-9,
    ):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict

        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)

        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.edge_weight_key = edge_weight_key
        self.edge_weight_mode = edge_weight_mode
        self.eps = float(eps)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = bool(use_norm)

        for _ in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))

        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    @staticmethod
    def _fallback_tensor(h: Dict[str, torch.Tensor]) -> torch.Tensor:
        if len(h) == 0:
            raise ValueError("h is empty; cannot infer device/dtype.")
        return next(iter(h.values()))

    def _dst_norm_edge_weight(self, sub: dgl.DGLHeteroGraph, ew: torch.Tensor) -> torch.Tensor:
        """
        ew: (E,1) float
        return ew_norm: (E,1) where sum over incoming edges to each dst node is 1
        """
        with sub.local_scope():
            sub.edata["_ew"] = ew
            sub.update_all(fn.copy_e("_ew", "m"), fn.sum("m", "_den"), etype=None)
            den = sub.dstdata["_den"]  # (num_dst,1)
            dst = sub.edges()[1]       # local dst idx
            den_e = den[dst].clamp_min(self.eps)
            return ew / den_e

    def forward(
        self,
        G: dgl.DGLHeteroGraph,
        h: Dict[str, torch.Tensor],
        return_attn: bool = False,
    ):
        """
        h: dict[ntype] -> (N_ntype, in_dim)
        returns new_h with same keys as G.ntypes (missing keys get zeros)
        """
        attn_dict = {} if return_attn else None

        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            fb = self._fallback_tensor(h)
            fb_device, fb_dtype = fb.device, fb.dtype

            # 1) per relation compute attn logits and messages
            for (srctype, etype, dsttype) in G.canonical_etypes:
                if (srctype not in h) or (dsttype not in h):
                    continue
                if (srctype, etype, dsttype) not in edge_dict:
                    continue

                sub = G[(srctype, etype, dsttype)]
                rid = edge_dict[(srctype, etype, dsttype)]

                k = self.k_linears[node_dict[srctype]](h[srctype]).view(-1, self.n_heads, self.d_k)
                v = self.v_linears[node_dict[srctype]](h[srctype]).view(-1, self.n_heads, self.d_k)
                q = self.q_linears[node_dict[dsttype]](h[dsttype]).view(-1, self.n_heads, self.d_k)

                rel_att = self.relation_att[rid]
                rel_pri = self.relation_pri[rid]
                rel_msg = self.relation_msg[rid]

                k = torch.einsum("nhd,hde->nhe", k, rel_att)
                v = torch.einsum("nhd,hde->nhe", v, rel_msg)

                sub.srcdata["k"] = k
                sub.dstdata["q"] = q
                sub.srcdata[f"v_{rid}"] = v

                sub.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn = sub.edata.pop("t").squeeze(-1)  # (E,H)

                # edge weight handling
                if self.edge_weight_mode != "none" and (self.edge_weight_key in sub.edata):
                    ew = sub.edata[self.edge_weight_key].float().unsqueeze(-1)  # (E,1)
                    ew = torch.clamp(ew, min=0.0)

                    if self.edge_weight_mode == "dst_norm":
                        ew = self._dst_norm_edge_weight(sub, ew)  # (E,1), dst-sum=1
                        # use as additive bias in log space is stable:
                        attn = attn + torch.log(ew + self.eps)  # (E,H)
                    elif self.edge_weight_mode == "mul_attn":
                        attn = attn * ew  # still will be normalized by softmax
                    else:
                        pass

                attn = attn * rel_pri / self.sqrt_dk
                attn = edge_softmax(sub, attn, norm_by="dst")  # (E,H)

                if return_attn:
                    attn_dict[(srctype, etype, dsttype)] = attn.detach().cpu()

                sub.edata["a"] = attn.unsqueeze(-1)  # (E,H,1)

            # 2) aggregate messages for computed relations
            funcs = {}
            for (srctype, etype, dsttype), rid in edge_dict.items():
                if (srctype, etype, dsttype) not in G.canonical_etypes:
                    continue
                if (srctype not in h) or (dsttype not in h):
                    continue
                funcs[(srctype, etype, dsttype)] = (
                    fn.u_mul_e(f"v_{rid}", "a", "m"),
                    fn.sum("m", "t"),
                )

            if len(funcs) > 0:
                G.multi_update_all(funcs, cross_reducer="mean")

            # 3) node update + skip
            new_h = {}
            for ntype in G.ntypes:
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])

                base = h.get(ntype, None)
                if base is None:
                    base = torch.zeros(
                        G.num_nodes(ntype), self.out_dim,
                        device=fb_device, dtype=fb_dtype
                    )

                if "t" in G.nodes[ntype].data:
                    t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                else:
                    t = torch.zeros(
                        G.num_nodes(ntype), self.out_dim,
                        device=base.device, dtype=base.dtype
                    )

                trans = self.drop(self.a_linears[n_id](t))
                out = trans * alpha + base * (1.0 - alpha)
                new_h[ntype] = self.norms[n_id](out) if self.use_norm else out

            if return_attn:
                return new_h, attn_dict
            return new_h
