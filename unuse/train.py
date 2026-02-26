# import numpy as np
# import torch
# from sksurv.metrics import concordance_index_censored

# from utils.pathway_contrast import PathwaySupConLoss


# @torch.no_grad()
# def calculate_risk(h: torch.Tensor):
#     hazards = torch.sigmoid(h)
#     survival = torch.cumprod(1 - hazards, dim=1)
#     risk = -torch.sum(survival, dim=1)
#     return risk.detach().cpu().numpy()


# @torch.no_grad()
# def eval_cindex(model, G, target_ntype, event_time, censorship, idx):
#     model.eval()
#     h = model(G, target_ntype)  # (N,K)
#     risk = calculate_risk(h[idx])

#     t = event_time[idx].detach().cpu().numpy()
#     c = censorship[idx].detach().cpu().numpy()

#     mask = np.isfinite(t) & np.isfinite(c)
#     if mask.sum() < 2:
#         return float("nan")

#     ev_bool = (1.0 - c[mask]).astype(bool)
#     return float(concordance_index_censored(ev_bool, t[mask], risk[mask], tied_tol=1e-8)[0])


# def inspect_pathway_maps(pathway_maps: dict, omics_ntypes, min_genes_per_pathway: int):
#     maps = pathway_maps.get("maps", None)
#     pnames = pathway_maps.get("pathway_names", None)
#     if maps is None or pnames is None:
#         print("[CL-DIAG][ERROR] pathway_maps missing keys: need 'maps' and 'pathway_names'")
#         return {"P": 0, "valid_ge2": 0}

#     P = len(pnames)
#     per_omics_ge_min = {}

#     for ntype in omics_ntypes:
#         cnt_ge_min = 0
#         if ntype in maps:
#             for p in range(P):
#                 idx = maps[ntype][p]
#                 if idx is not None and idx.numel() >= min_genes_per_pathway:
#                     cnt_ge_min += 1
#         per_omics_ge_min[ntype] = cnt_ge_min

#     valid_ge2 = 0
#     for p in range(P):
#         k = 0
#         for ntype in omics_ntypes:
#             if ntype not in maps:
#                 continue
#             idx = maps[ntype][p]
#             if idx is not None and idx.numel() >= min_genes_per_pathway:
#                 k += 1
#         if k >= 2:
#             valid_ge2 += 1

#     print(f"[CL-DIAG] P={P}, min_genes={min_genes_per_pathway}, valid_pathways_ge2omics={valid_ge2}")
#     for ntype in omics_ntypes:
#         print(f"[CL-DIAG] {ntype}: pathways_ge_min={per_omics_ge_min[ntype]}")
#     if valid_ge2 == 0:
#         print(
#             "[CL-DIAG][WARN] valid_pathways_ge2omics==0 => CL will be 0.\n"
#             "Likely gene-name mismatch (GMT uses gene symbols). Map CNV/Methy IDs to symbols then rebuild pathway_maps.pt.\n"
#             "You can also try lowering min_genes_per_pathway to 1~3."
#         )
#     return {"P": P, "valid_ge2": valid_ge2}


# def train_one_fold(
#     model,
#     G,
#     target_ntype,
#     y_disc,
#     event_time,
#     censorship,
#     train_idx,
#     val_idx,
#     test_idx,
#     loss_fn,
#     optimizer,
#     scheduler,
#     n_epoch: int,
#     eval_every: int,
#     clip: float,
#     # --------- contrastive ---------
#     pathway_maps: dict = None,
#     cl_module=None,                     # ✅ NEW: 由外部创建并传入（或None）
#     lambda_cl: float = 0.1,
#     # --------- patch attn export ---------
#     return_patch_attn_every: int = 0,
# ):
#     device = next(model.parameters()).device

#     best_val = -1.0
#     best_test = -1.0
#     best_epoch = -1
#     best_state = None
#     last_patch_attn = None

#     for epoch in range(1, n_epoch + 1):
#         model.train()
#         if cl_module is not None:
#             cl_module.train()

#         if return_patch_attn_every > 0 and (epoch % return_patch_attn_every == 0):
#             out = model(G, target_ntype, return_patch_attn=True, return_omics_h=True)
#             h, patch_attn, omics_h = out
#             last_patch_attn = patch_attn.detach().cpu()
#         else:
#             out = model(G, target_ntype, return_omics_h=True)
#             h, omics_h = out

#         loss_surv_raw = loss_fn(h=h[train_idx], y=y_disc[train_idx], t=event_time[train_idx], c=censorship[train_idx])
#         loss_surv = loss_surv_raw / max(1, train_idx.numel())

#         if (cl_module is not None) and (pathway_maps is not None) and (lambda_cl > 0):
#             loss_cl = cl_module(omics_h=omics_h, pathway_maps=pathway_maps, device=device)
#         else:
#             loss_cl = torch.zeros((), device=device)

#         loss = loss_surv #+ float(lambda_cl) * loss_cl

#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()

#         params = list(model.parameters())
#         if cl_module is not None:
#             params += list(cl_module.parameters())
#         torch.nn.utils.clip_grad_norm_(params, clip)

#         optimizer.step()
#         scheduler.step()   # ✅ 现在不会再 KeyError

#         if epoch % eval_every == 0 or epoch == 1 or epoch == n_epoch:
#             tr = eval_cindex(model, G, target_ntype, event_time, censorship, train_idx)
#             va = eval_cindex(model, G, target_ntype, event_time, censorship, val_idx)
#             te = eval_cindex(model, G, target_ntype, event_time, censorship, test_idx)

#             if not np.isnan(va) and va > best_val:
#                 best_val = va
#                 best_test = te
#                 best_epoch = epoch
#                 best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
#                 if cl_module is not None:
#                     best_state["_cl_module"] = {k: v.detach().cpu().clone() for k, v in cl_module.state_dict().items()}

#             lr = optimizer.param_groups[0]["lr"]
#             print(
#                 f"Epoch {epoch:4d} | LR {lr:.6f} | "
#                 f"Loss {loss.item():.4f} (Surv {loss_surv.item():.4f}, CL {loss_cl.item():.4f}, λ={lambda_cl}) | "
#                 f"C-index Train {tr:.4f} | Val {va:.4f} (Best {best_val:.4f}@{best_epoch}) | "
#                 f"Test {te:.4f} (Best {best_test:.4f})"
#             )

#     final_test = eval_cindex(model, G, target_ntype, event_time, censorship, test_idx)
#     return {
#         "best_val": float(best_val),
#         "best_test_tracked": float(best_test),
#         "best_epoch": int(best_epoch),
#         "final_test": float(final_test),
#         "best_state": best_state,
#         "last_patch_attn": last_patch_attn,
#     }

# utils/train.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import copy
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sksurv.metrics import concordance_index_censored


def _risk_from_logits(h: torch.Tensor) -> torch.Tensor:
    """
    SurvPath-style:
      hazards = sigmoid(h)
      survival = cumprod(1 - hazards)
      risk = -sum(survival)
    """
    hazards = torch.sigmoid(h)
    surv = torch.cumprod(1.0 - hazards, dim=1)
    risk = -torch.sum(surv, dim=1)
    return risk


@torch.no_grad()
def _eval_cindex(
    logits: torch.Tensor,
    event_time: torch.Tensor,
    censorship: torch.Tensor,
    idx: torch.Tensor,
) -> float:
    """
    censorship: 1=censored, 0=event
    sksurv needs event indicator: event=True if observed event
    """
    if idx.numel() == 0:
        return float("nan")
    risk = _risk_from_logits(logits[idx]).detach().cpu().numpy().astype(np.float64)
    t = event_time[idx].detach().cpu().numpy().astype(np.float64)
    c = censorship[idx].detach().cpu().numpy().astype(np.int64)
    event = (c == 0)
    ci = concordance_index_censored(event, t, risk)[0]
    return float(ci)


def train_one_fold(
    model: nn.Module,
    G,
    target_ntype: str,
    y_disc: torch.Tensor,
    event_time: torch.Tensor,
    censorship: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    n_epoch: int = 200,
    eval_every: int = 5,
    clip: float = 1.0,

    # ---- pathway CL (your existing) ----
    pathway_maps: Optional[dict] = None,
    cl_module: Optional[nn.Module] = None,
    lambda_cl: float = 0.0,

    # ---- ✅ cluster constraint (new) ----
    cluster_maps: Optional[dict] = None,
    cluster_module: Optional[nn.Module] = None,
    lambda_cluster: float = 0.0,
    cluster_ntypes: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Returns metrics dict with:
      best_val, best_test_tracked, final_test, best_epoch, best_state
    """
    best_val = -1.0
    best_test_tracked = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, int(n_epoch) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # We need omics embeddings to compute CL losses
        out = model(G, target_ntype, return_omics_h=True)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            logits, omics_h = out[0], out[-1]
        else:
            logits, omics_h = out, None

        # survival loss
        loss_surv = loss_fn(
            logits[train_idx],
            y_disc[train_idx],
            event_time[train_idx],
            censorship[train_idx],
        )

        loss = loss_surv

        # ---- pathway CL (keep your original behavior) ----
        loss_pathway = None
        if (cl_module is not None) and (pathway_maps is not None) and (lambda_cl is not None) and (lambda_cl > 0) and (omics_h is not None):
            try:
                loss_pathway = cl_module(omics_h, pathway_maps)
                loss = loss + float(lambda_cl) * loss_pathway
            except Exception:
                # be robust: if some ablation removed gene types
                loss_pathway = None

        # ---- ✅ cluster constraint ----
        loss_cluster = None
        if (cluster_module is not None) and (cluster_maps is not None) and (lambda_cluster is not None) and (lambda_cluster > 0) and (omics_h is not None):
            try:
                loss_cluster = cluster_module(omics_h, cluster_maps, ntypes=cluster_ntypes)
                loss = loss + float(lambda_cluster) * loss_cluster
            except Exception:
                loss_cluster = None

        loss.backward()
        if clip is not None and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # evaluation
        if (epoch % int(eval_every) == 0) or (epoch == n_epoch):
            model.eval()
            with torch.no_grad():
                logits_eval = model(G, target_ntype)  # no need embeddings here

            val_ci = _eval_cindex(logits_eval, event_time, censorship, val_idx)
            test_ci = _eval_cindex(logits_eval, event_time, censorship, test_idx)

            if val_ci > best_val:
                best_val = val_ci
                best_test_tracked = test_ci
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            # optional print
            msg = f"[E{epoch:03d}] loss={float(loss.item()):.4f} surv={float(loss_surv.item()):.4f} val_ci={val_ci:.4f} test_ci={test_ci:.4f}"
            if loss_pathway is not None:
                msg += f" | pCL={float(loss_pathway.item()):.4f}*{lambda_cl}"
            if loss_cluster is not None:
                msg += f" | cCL={float(loss_cluster.item()):.4f}*{lambda_cluster}"
            print(msg)

    # final eval
    model.eval()
    with torch.no_grad():
        logits_final = model(G, target_ntype)
    final_test = _eval_cindex(logits_final, event_time, censorship, test_idx)

    return {
        "best_val": float(best_val),
        "best_test_tracked": float(best_test_tracked),
        "final_test": float(final_test),
        "best_epoch": int(best_epoch),
        "best_state": best_state,
    }
