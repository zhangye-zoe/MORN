
# utils/walk_sampler.py
import random
from typing import Dict, List, Tuple, Optional

import torch
import dgl


class TypeConstrainedWalkSampler:
    """
    Type-constrained random walk on a DGL heterograph.

    We sample walks on node IDs (per ntype).
    Each step chooses next ntype based on constraints, then samples ONE neighbor along a
    valid canonical etype (src_t, etype, dst_t).

    This implementation is compatible with older DGL versions where
    dgl.sampling.sample_neighbors does NOT accept `etype=` keyword argument.
    """

    def __init__(
        self,
        G: dgl.DGLHeteroGraph,
        allowed_next_types: Optional[Dict[str, List[str]]] = None,
        metapath_types_list: Optional[List[List[str]]] = None,
        edge_dir: str = "out",
        prob_key: Optional[str] = None,   # e.g. "w" for weighted sampling
        seed: int = 0,
    ):
        """
        Args:
            G: heterograph
            allowed_next_types: dict[src_ntype] -> list of allowed dst ntypes
            metapath_types_list: list of metapaths, each is a list of ntypes
                e.g. ["patient","gene_mRNA","gene_miRNA","gene_mRNA"]
                If provided, we sample a metapath randomly each walk and follow its type sequence.
            edge_dir: "out" or "in"
            prob_key: edge feature key for sampling probability (optional)
        """
        self.G = G
        self.allowed_next_types = allowed_next_types or {}
        self.metapath_types_list = metapath_types_list
        self.edge_dir = edge_dir
        self.prob_key = prob_key
        random.seed(seed)

        # build adjacency of canonical etypes by (src_t, dst_t)
        self._pair2etypes: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
        for et in G.canonical_etypes:
            s, _, d = et
            self._pair2etypes.setdefault((s, d), []).append(et)

    def _pick_next_type(self, cur_t: str) -> Optional[str]:
        """
        Pick next ntype from allowed_next_types if provided.
        If not provided for this type, fallback to any reachable dst type in graph.
        """
        if cur_t in self.allowed_next_types and len(self.allowed_next_types[cur_t]) > 0:
            return random.choice(self.allowed_next_types[cur_t])

        # fallback: pick any dst type that has at least one etype from cur_t
        cands = []
        for (s, d), ets in self._pair2etypes.items():
            if s == cur_t and len(ets) > 0:
                cands.append(d)
        if len(cands) == 0:
            return None
        return random.choice(cands)

    def _sample_one_step(self, src_t: str, src_id: int, dst_t: str) -> Optional[int]:
        """
        Sample ONE neighbor id of type dst_t from (src_t=src_id) with fanout=1.

        IMPORTANT: We do NOT pass `etype=` kw to sample_neighbors.
        Instead we take a single-etype subgraph self.G[canonical_etype] and sample on it.
        """
        etypes = self._pair2etypes.get((src_t, dst_t), [])
        if len(etypes) == 0:
            return None

        # pick one canonical etype (if multiple exist)
        et = random.choice(etypes)

        # single-etype subgraph (bipartite heterograph)
        sub = self.G[et]

        seed_nodes = {src_t: torch.tensor([src_id], dtype=torch.int64)}
        # In older DGL, signature: sample_neighbors(g, nodes, fanout, edge_dir='in', prob=None, replace=False)
        # Use prob feature if exists & prob_key is provided
        prob = self.prob_key if (self.prob_key is not None and self.prob_key in sub.edges[et].data) else None

        frontier = dgl.sampling.sample_neighbors(
            sub,
            seed_nodes,
            fanout=1,
            edge_dir=self.edge_dir,
            prob=prob,
            replace=False,
        )

        # frontier contains only this etype
        # get edges of this etype in sampled frontier
        u, v = frontier.edges(etype=et)
        if u.numel() == 0:
            return None

        # depending on edge_dir, decide which side is next node
        # edge_dir="out": from src -> dst, so v is dst
        # edge_dir="in" : from dst -> src, so u is dst (because sampled incoming neighbors to src)
        if self.edge_dir == "out":
            nxt = int(v[0].item())
        else:
            nxt = int(u[0].item())

        return nxt

    def sample_walk_node_ids(
        self,
        start_ntype: str,
        start_node_ids: torch.Tensor,
        walk_len: int,
        walks_per_node: int,
    ) -> List[List[Tuple[str, int]]]:
        """
        Returns:
            list of walks; each walk is a list of (ntype, node_id)
        """
        start_node_ids = start_node_ids.detach().cpu().long().tolist()
        walks: List[List[Tuple[str, int]]] = []

        for sid in start_node_ids:
            for _ in range(walks_per_node):
                if self.metapath_types_list is not None and len(self.metapath_types_list) > 0:
                    types = random.choice(self.metapath_types_list)
                    # enforce start type
                    if types[0] != start_ntype:
                        types = [start_ntype] + types[1:]
                    # walk_len means number of steps; types length should be walk_len+1
                    if len(types) < walk_len + 1:
                        # repeat last type if too short
                        types = types + [types[-1]] * (walk_len + 1 - len(types))
                    types = types[: walk_len + 1]
                else:
                    types = None

                cur_t = start_ntype
                cur_id = int(sid)
                walk = [(cur_t, cur_id)]

                for step in range(walk_len):
                    if types is not None:
                        nxt_t = types[step + 1]
                    else:
                        nxt_t = self._pick_next_type(cur_t)
                        if nxt_t is None:
                            break

                    nxt_id = self._sample_one_step(cur_t, cur_id, nxt_t)
                    if nxt_id is None:
                        break

                    walk.append((nxt_t, nxt_id))
                    cur_t, cur_id = nxt_t, nxt_id

                walks.append(walk)

        return walks



    # """
    # Type-constrained random walk starting from patient nodes.

    # Key idea:
    # - We walk over NODE TYPES with biological constraints.
    # - At each step, we pick a next NODE TYPE from allowed transitions,
    #   then pick one CANONICAL ETYPE that matches (src_type -> dst_type),
    #   then sample a neighbor in that etype.

    # This does NOT use patient as message source; it's only for CL sampling.

    # You can provide:
    # - allowed_next_types: dict[src_type] = list[next_type]
    #   (e.g., patient -> gene_mRNA; gene_mRNA -> gene_miRNA or gene_Methy etc.)
    # OR
    # - metapath_types_list: list of node-type sequences, e.g.
    #   [
    #     ["patient","gene_mRNA","gene_miRNA","gene_mRNA"],
    #     ["patient","gene_mRNA","gene_Methy","gene_mRNA"],
    #   ]
    # """

    # def __init__(
    #     self,
    #     G: dgl.DGLHeteroGraph,
    #     start_ntype: str = "patient",
    #     allowed_next_types: Optional[Dict[str, List[str]]] = None,
    #     metapath_types_list: Optional[List[List[str]]] = None,
    #     etype_sampling: str = "uniform",  # "uniform" or "weighted_by_edge_weight"
    #     edge_weight_key: str = "w",
    # ):
    #     self.G = G
    #     self.start_ntype = start_ntype
    #     self.allowed_next_types = allowed_next_types
    #     self.metapath_types_list = metapath_types_list
    #     self.etype_sampling = etype_sampling
    #     self.edge_weight_key = edge_weight_key

    #     # build map: (src_type, dst_type) -> list[canonical_etype]
    #     self.sd2etypes: Dict[Tuple[str, str], List[CanonicalEType]] = {}
    #     for (s, e, d) in G.canonical_etypes:
    #         self.sd2etypes.setdefault((s, d), []).append((s, e, d))

    #     # if user didn't provide allowed_next_types and didn't provide metapath list,
    #     # we build a conservative default: from each type can go to any neighbor type in graph,
    #     # but we still forbid "patient -> patient" by default
    #     if self.allowed_next_types is None and self.metapath_types_list is None:
    #         self.allowed_next_types = {}
    #         for (s, _, d) in G.canonical_etypes:
    #             if s == "patient" and d == "patient":
    #                 continue
    #             self.allowed_next_types.setdefault(s, [])
    #             if d not in self.allowed_next_types[s]:
    #                 self.allowed_next_types[s].append(d)

    # def _pick_etype(self, src_t: str, dst_t: str) -> Optional[CanonicalEType]:
    #     cands = self.sd2etypes.get((src_t, dst_t), [])
    #     if len(cands) == 0:
    #         return None
    #     if self.etype_sampling == "uniform":
    #         return random.choice(cands)

    #     # weighted_by_edge_weight: choose etype proportional to sum of edge weights (if available)
    #     scores = []
    #     for et in cands:
    #         if self.edge_weight_key in self.G.edges[et].data:
    #             w = self.G.edges[et].data[self.edge_weight_key].float()
    #             scores.append(float(w.sum().item()))
    #         else:
    #             scores.append(1.0)
    #     tot = sum(scores)
    #     if tot <= 0:
    #         return random.choice(cands)
    #     r = random.random() * tot
    #     acc = 0.0
    #     for et, sc in zip(cands, scores):
    #         acc += sc
    #         if r <= acc:
    #             return et
    #     return cands[-1]

    # def _sample_one_step(self, src_t: str, src_id: int, dst_t: str) -> Optional[int]:
    #     et = self._pick_etype(src_t, dst_t)
    #     if et is None:
    #         return None

    #     # sample one neighbor
    #     # NOTE: DGL sample_neighbors expects a dict for hetero graphs
    #     frontier = dgl.sampling.sample_neighbors(self.G, {src_t: torch.tensor([src_id], device="cpu")}, fanout=1, etype=et)
    #     u, v = frontier.edges(etype=et)
    #     if len(v) == 0:
    #         return None
    #     # v is dst node IDs in frontier's ID space == original ID space (since sample_neighbors keeps IDs)
    #     return int(v[0].item())

    # def sample_walk_node_ids(
    #     self,
    #     start_ids: torch.Tensor,
    #     walk_len: int = 3,
    #     num_walks_per_patient: int = 4,
    # ) -> Dict[int, List[List[Tuple[str, int]]]]:
    #     """
    #     Returns:
    #       walks[pid] = list of walks
    #       each walk is a list of (ntype, nid), starting with ("patient", pid)
    #     """
    #     start_ids = start_ids.detach().cpu().long().tolist()
    #     out: Dict[int, List[List[Tuple[str, int]]]] = {}

    #     for pid in start_ids:
    #         pid_walks = []
    #         for _ in range(num_walks_per_patient):
    #             if self.metapath_types_list is not None:
    #                 types = random.choice(self.metapath_types_list)
    #                 # enforce start
    #                 if len(types) == 0 or types[0] != self.start_ntype:
    #                     types = [self.start_ntype] + types
    #                 # cut / pad
    #                 types = types[: (walk_len + 1)]
    #                 if len(types) < (walk_len + 1):
    #                     # if too short, just stop early
    #                     pass
    #             else:
    #                 # type-walk by allowed transitions
    #                 types = [self.start_ntype]
    #                 cur = self.start_ntype
    #                 for _k in range(walk_len):
    #                     nxts = self.allowed_next_types.get(cur, [])
    #                     if len(nxts) == 0:
    #                         break
    #                     nxt = random.choice(nxts)
    #                     types.append(nxt)
    #                     cur = nxt

    #             # now sample node IDs following types
    #             walk = [(self.start_ntype, pid)]
    #             cur_t, cur_id = self.start_ntype, pid
    #             ok = True
    #             for step in range(1, len(types)):
    #                 nxt_t = types[step]
    #                 nxt_id = self._sample_one_step(cur_t, cur_id, nxt_t)
    #                 if nxt_id is None:
    #                     ok = False
    #                     break
    #                 walk.append((nxt_t, nxt_id))
    #                 cur_t, cur_id = nxt_t, nxt_id

    #             if ok and len(walk) >= 2:
    #                 pid_walks.append(walk)

    #         out[pid] = pid_walks
    #     return out

    # @staticmethod
    # def aggregate_walk_embeddings(
    #     G: dgl.DGLHeteroGraph,
    #     h_dict: Dict[str, torch.Tensor],
    #     walks: Dict[int, List[List[Tuple[str, int]]]],
    #     agg: str = "mean",
    # ) -> torch.Tensor:
    #     """
    #     Convert per-patient walks into a patient embedding tensor (B,D).
    #     Uses node embeddings from h_dict (e.g. gene embeddings after omics encoder).
    #     """
    #     # infer embedding dim/device
    #     any_t = next(iter(h_dict.values()))
    #     device = any_t.device
    #     D = any_t.size(-1)

    #     pids = list(walks.keys())
    #     B = len(pids)
    #     out = torch.zeros((B, D), device=device, dtype=any_t.dtype)

    #     for i, pid in enumerate(pids):
    #         all_vecs = []
    #         for w in walks[pid]:
    #             vecs = []
    #             for (nt, nid) in w:
    #                 if nt not in h_dict:
    #                     continue
    #                 vecs.append(h_dict[nt][nid])
    #             if len(vecs) > 0:
    #                 w_emb = torch.stack(vecs, dim=0).mean(dim=0)
    #                 all_vecs.append(w_emb)
    #         if len(all_vecs) == 0:
    #             continue
    #         all_vecs = torch.stack(all_vecs, dim=0)
    #         out[i] = all_vecs.mean(dim=0) if agg == "mean" else all_vecs.sum(dim=0)

    #     return out, torch.tensor(pids, device=device, dtype=torch.long)
