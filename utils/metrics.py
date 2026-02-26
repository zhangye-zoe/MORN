import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

@torch.no_grad()
def calculate_risk(h: torch.Tensor):
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1)
    return risk.detach().cpu().numpy(), survival.detach().cpu().numpy()

@torch.no_grad()
def eval_cindex(model, G, target_ntype, event_time, censorship, idx):
    model.eval()
    h = model(G, target_ntype)  # (N,K)
    risk, _ = calculate_risk(h[idx])

    t = event_time[idx].detach().cpu().numpy()
    c = censorship[idx].detach().cpu().numpy()

    mask = np.isfinite(t) & np.isfinite(c)
    if mask.sum() < 2:
        return float("nan")

    ev_bool = (1.0 - c[mask]).astype(bool)
    cind = concordance_index_censored(ev_bool, t[mask], risk[mask], tied_tol=1e-8)[0]
    return float(cind)
