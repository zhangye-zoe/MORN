import json
from pathlib import Path
import torch
import dgl

@torch.no_grad()
def dump_all_attn_to_files(
    G_cpu: dgl.DGLHeteroGraph,
    all_attn,  # list[dict], canonical_etype -> (E,H) CPU
    out_dir: str,
    meta: dict,
    edge_weight_key: str = "w",
    save_edge_weight: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(all_attn)
    example = None
    for d in all_attn:
        if d:
            example = next(iter(d.values()))
            break
    n_heads = int(example.shape[1]) if example is not None else None

    manifest = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "canonical_etypes": [list(x) for x in G_cpu.canonical_etypes],
        "files": [],
        "meta": meta,
    }

    for li in range(n_layers):
        layer_attn = all_attn[li]
        for c_etype in G_cpu.canonical_etypes:
            if c_etype not in layer_attn:
                continue

            sub = G_cpu[c_etype]
            src, dst = sub.edges(order="eid")

            attn_head = layer_attn[c_etype].contiguous().to(torch.float32)
            attn_mean = attn_head.mean(dim=1).contiguous()

            save_obj = {
                "canonical_etype": c_etype,
                "layer": li,
                "src": src.to(torch.int64),
                "dst": dst.to(torch.int64),
                "attn_head": attn_head,
                "attn_mean": attn_mean,
            }

            if save_edge_weight and (edge_weight_key in sub.edata):
                save_obj["edge_w"] = sub.edata[edge_weight_key].to(torch.float32).contiguous()

            fname = f"attn_layer{li}__{c_etype[0]}__{c_etype[1]}__{c_etype[2]}.pt"
            fpath = out_dir / fname
            torch.save(save_obj, fpath)

            manifest["files"].append({
                "layer": li,
                "canonical_etype": list(c_etype),
                "path": str(fpath),
                "num_edges": int(sub.num_edges()),
            })

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] attention dump -> {out_dir}")
