import torch
import dgl

def ensure_nid(G: dgl.DGLHeteroGraph):
    for ntype in G.ntypes:
        if "nid" not in G.nodes[ntype].data:
            G.nodes[ntype].data["nid"] = torch.arange(G.num_nodes(ntype), dtype=torch.long)

def ensure_edge_weight(G: dgl.DGLHeteroGraph, key: str):
    for c_etype in G.canonical_etypes:
        if key not in G.edges[c_etype].data:
            G.edges[c_etype].data[key] = torch.ones(G.num_edges(c_etype), dtype=torch.float32)

def count_params(model):
    return sum(p.numel() for p in model.parameters())


