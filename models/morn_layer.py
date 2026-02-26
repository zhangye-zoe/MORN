# import math
# from typing import Dict, Tuple

# import dgl
# import dgl.function as fn
# import torch
# import torch.nn as nn
# from dgl.nn.functional import edge_softmax


# class MORNLayer(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         node_dict: Dict[str, int],
#         edge_dict: Dict[Tuple[str, str, str], int],
#         n_heads: int,
#         dropout: float = 0.2,
#         use_norm: bool = False,
#         edge_weight_key: str = "w",
#         edge_weight_mode: str = "mul_attn",  # "mul_attn" or "none"
#     ):
#         super().__init__()
#         assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"

#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.node_dict = node_dict
#         self.edge_dict = edge_dict

#         self.num_types = len(node_dict)
#         self.num_relations = len(edge_dict)

#         self.n_heads = n_heads
#         self.d_k = out_dim // n_heads
#         self.sqrt_dk = math.sqrt(self.d_k)

#         self.edge_weight_key = edge_weight_key
#         self.edge_weight_mode = edge_weight_mode

#         self.k_linears = nn.ModuleList()
#         self.q_linears = nn.ModuleList()
#         self.v_linears = nn.ModuleList()
#         self.a_linears = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         self.use_norm = use_norm

#         for _ in range(self.num_types):
#             self.k_linears.append(nn.Linear(in_dim, out_dim))
#             self.q_linears.append(nn.Linear(in_dim, out_dim))
#             self.v_linears.append(nn.Linear(in_dim, out_dim))
#             self.a_linears.append(nn.Linear(out_dim, out_dim))
#             if use_norm:
#                 self.norms.append(nn.LayerNorm(out_dim))

#         self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
#         self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
#         self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))

#         self.skip = nn.Parameter(torch.ones(self.num_types))
#         self.drop = nn.Dropout(dropout)

#         nn.init.xavier_uniform_(self.relation_att)
#         nn.init.xavier_uniform_(self.relation_msg)

#     def forward(self, G: dgl.DGLHeteroGraph, h: Dict[str, torch.Tensor], return_attn: bool = False):
#         """
#         return_attn=True 时，额外返回：
#           attn_dict[(srctype, etype, dsttype)] = Tensor(E, H) (softmax后，CPU)
#         """
#         attn_dict = {} if return_attn else None

#         with G.local_scope():
#             node_dict, edge_dict = self.node_dict, self.edge_dict

#             for (srctype, etype, dsttype) in G.canonical_etypes:
#                 sub = G[(srctype, etype, dsttype)]
#                 rid = edge_dict[(srctype, etype, dsttype)]

#                 k_linear = self.k_linears[node_dict[srctype]]
#                 v_linear = self.v_linears[node_dict[srctype]]
#                 q_linear = self.q_linears[node_dict[dsttype]]

#                 k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
#                 v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
#                 q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

#                 rel_att = self.relation_att[rid]
#                 rel_pri = self.relation_pri[rid]
#                 rel_msg = self.relation_msg[rid]

#                 k = torch.einsum("nhd,hde->nhe", k, rel_att)
#                 v = torch.einsum("nhd,hde->nhe", v, rel_msg)

#                 sub.srcdata["k"] = k
#                 sub.dstdata["q"] = q
#                 sub.srcdata[f"v_{rid}"] = v

#                 sub.apply_edges(fn.v_dot_u("q", "k", "t"))
#                 attn = sub.edata.pop("t").squeeze(-1)

#                 if self.edge_weight_mode == "mul_attn" and self.edge_weight_key in sub.edata:
#                     ew = sub.edata[self.edge_weight_key].float().unsqueeze(-1)  # (E,1)
#                     attn = attn * ew

#                 attn = attn * rel_pri / self.sqrt_dk
#                 attn = edge_softmax(sub, attn, norm_by="dst")  # (E,H)

#                 if return_attn:
#                     attn_dict[(srctype, etype, dsttype)] = attn.detach().cpu()

#                 sub.edata["t"] = attn.unsqueeze(-1)  # (E,H,1)

#             funcs = {}
#             for (srctype, etype, dsttype), rid in edge_dict.items():
#                 if (srctype, etype, dsttype) not in G.canonical_etypes:
#                     continue
#                 funcs[(srctype, etype, dsttype)] = (
#                     fn.u_mul_e(f"v_{rid}", "t", "m"),
#                     fn.sum("m", "t"),
#                 )

#             G.multi_update_all(funcs, cross_reducer="mean")

#             new_h = {}
#             for ntype in G.ntypes:
#                 n_id = node_dict[ntype]
#                 alpha = torch.sigmoid(self.skip[n_id])

#                 if "t" in G.nodes[ntype].data:
#                     t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
#                 else:
#                     t = torch.zeros(G.num_nodes(ntype), self.out_dim, device=h[ntype].device)

#                 trans = self.drop(self.a_linears[n_id](t))
#                 out = trans * alpha + h[ntype] * (1 - alpha)
#                 new_h[ntype] = self.norms[n_id](out) if self.use_norm else out

#             if return_attn:
#                 return new_h, attn_dict
#             return new_h


# models/morn_layer.py
import math
from typing import Dict, Tuple, Optional

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax


class MORNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        node_dict: Dict[str, int],
        edge_dict: Dict[Tuple[str, str, str], int],
        n_heads: int,
        dropout: float = 0.2,
        use_norm: bool = False,
        edge_weight_key: str = "w",
        edge_weight_mode: str = "mul_attn",  # "mul_attn" or "none"
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

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for _ in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        # relation-specific parameters
        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))

        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    @staticmethod
    def _fallback_tensor(h: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Pick any tensor from h to infer device/dtype.
        """
        if len(h) == 0:
            raise ValueError("Input h is empty; cannot infer device/dtype.")
        return next(iter(h.values()))

    def forward(
        self,
        G: dgl.DGLHeteroGraph,
        h: Dict[str, torch.Tensor],
        return_attn: bool = False,
    ):
        """
        G: DGLHeteroGraph
        h: dict[ntype] -> Tensor[num_nodes(ntype), in_dim]
        return_attn=True 时，额外返回：
          attn_dict[(srctype, etype, dsttype)] = Tensor(E, H) (softmax后，CPU)

        ✅ Robust to ablations:
        - If some ntype exists in G.ntypes but missing in h (e.g. "wsi" disabled),
          this layer will create zero base embeddings for that ntype in the output.
        - If some canonical relation involves missing ntypes in h, we skip that relation.
        """
        attn_dict = {} if return_attn else None

        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict

            # fallback device/dtype
            fb = self._fallback_tensor(h)
            fb_device, fb_dtype = fb.device, fb.dtype

            # 1) compute attention for EACH CANONICAL relation
            #    skip relations whose srctype/dsttype missing in h (ablation-safe)
            for (srctype, etype, dsttype) in G.canonical_etypes:
                if (srctype not in h) or (dsttype not in h):
                    # e.g. wsi was disabled so h has no "wsi" but G still contains it
                    continue

                if (srctype, etype, dsttype) not in edge_dict:
                    continue

                sub = G[(srctype, etype, dsttype)]
                rid = edge_dict[(srctype, etype, dsttype)]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                rel_att = self.relation_att[rid]
                rel_pri = self.relation_pri[rid]
                rel_msg = self.relation_msg[rid]

                # (N,H,d) x (H,d,d) -> (N,H,d)
                k = torch.einsum("nhd,hde->nhe", k, rel_att)
                v = torch.einsum("nhd,hde->nhe", v, rel_msg)

                sub.srcdata["k"] = k
                sub.dstdata["q"] = q
                sub.srcdata[f"v_{rid}"] = v

                sub.apply_edges(fn.v_dot_u("q", "k", "t"))  # (E,H,1)
                attn = sub.edata.pop("t").squeeze(-1)       # (E,H)

                if self.edge_weight_mode == "mul_attn" and (self.edge_weight_key in sub.edata):
                    ew = sub.edata[self.edge_weight_key].float().unsqueeze(-1)  # (E,1)
                    attn = attn * ew  # broadcast to (E,H)

                attn = attn * rel_pri / self.sqrt_dk
                attn = edge_softmax(sub, attn, norm_by="dst")  # (E,H)

                if return_attn:
                    attn_dict[(srctype, etype, dsttype)] = attn.detach().cpu()

                sub.edata["t"] = attn.unsqueeze(-1)  # (E,H,1)

            # 2) aggregate messages for relations we actually computed (those with v_{rid} and t)
            funcs = {}
            for (srctype, etype, dsttype), rid in edge_dict.items():
                if (srctype, etype, dsttype) not in G.canonical_etypes:
                    continue
                if (srctype not in h) or (dsttype not in h):
                    continue  # we skipped computing this relation above
                # if relation computed, then subgraph has srcdata[f"v_{rid}"] and edata["t"]
                funcs[(srctype, etype, dsttype)] = (
                    fn.u_mul_e(f"v_{rid}", "t", "m"),
                    fn.sum("m", "t"),
                )

            if len(funcs) > 0:
                G.multi_update_all(funcs, cross_reducer="mean")
            # else: no message passing this layer (all relations skipped)

            # 3) node update + skip
            new_h = {}
            for ntype in G.ntypes:
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])

                # ✅ base embedding: if missing in h (ablation), use zeros
                base = h.get(ntype, None)
                if base is None:
                    base = torch.zeros(
                        G.num_nodes(ntype),
                        self.out_dim,
                        device=fb_device,
                        dtype=fb_dtype,
                    )

                if "t" in G.nodes[ntype].data:
                    t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                else:
                    t = torch.zeros(
                        G.num_nodes(ntype),
                        self.out_dim,
                        device=base.device,
                        dtype=base.dtype,
                    )

                trans = self.drop(self.a_linears[n_id](t))
                out = trans * alpha + base * (1.0 - alpha)
                new_h[ntype] = self.norms[n_id](out) if self.use_norm else out

            if return_attn:
                return new_h, attn_dict
            return new_h
