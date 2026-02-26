import os
import torch
import torch.nn as nn
import pandas as pd

class NLLSurvLoss(nn.Module):
    """
    Discrete-time NLL (SurvPath / Patch-GCN style)
    hazards = sigmoid(h)
    y: (N,) bin index in [0, K-1]
    c: (N,) 1=censored, 0=event
    """
    def __init__(self, alpha: float = 0.0, reduction: str = "sum", eps: float = 1e-7):
        super().__init__()
        self.alpha = float(alpha)
        assert reduction in ("sum", "mean")
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, h: torch.Tensor, y: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # t is unused (kept for interface compatibility)
        hazards = torch.sigmoid(h).clamp(self.eps, 1.0 - self.eps)  # (N,K)
        # print('harzards', hazards.shape, hazards)
        N, K = hazards.shape

        # print('y', y.shape, y)
        # print('c', c.shape, c)

        y = y.view(-1).long()
        c = c.view(-1).float()

        # safer than clamp: if you really want clamp, at least warn
        if (y < 0).any() or (y >= K).any():
            raise ValueError(f"y out of range: min={int(y.min())}, max={int(y.max())}, K={K}")

        # S_inclusive[:,k] = prod_{j<=k} (1-h_j) = S(k)
        S_inclusive = torch.cumprod(1.0 - hazards, dim=1)           # (N,K)
        # print('S_inclusive', S_inclusive.shape, S_inclusive)

        # S_before[:,k] = S(k-1), with S_before[:,0]=1
        S_before = torch.cat([torch.ones(N, 1, device=h.device), S_inclusive[:, :-1]], dim=1)  # (N,K)
        # print('S_before', S_before.shape, S_before)

        idx = y.unsqueeze(1)  # (N,1)
        S_before_y = torch.gather(S_before, 1, idx).squeeze(1)      # S(y-1)
        h_y        = torch.gather(hazards, 1, idx).squeeze(1)       # h(y)
        S_y        = torch.gather(S_inclusive, 1, idx).squeeze(1)   # S(y) = S(y-1)*(1-h(y))

        # event:  log P(T=y) = log S(y-1) + log h(y)
        uncensored = -(torch.log(S_before_y) + torch.log(h_y))

        # cens:   log P(T>y) = log S(y)
        censored  = -torch.log(S_y)

        neg_l = (1.0 - c) * uncensored + c * censored

        # alpha reweight (as in Patch-GCN code)
        if self.alpha is not None and self.alpha > 0:
            loss = (1.0 - self.alpha) * neg_l + self.alpha * ((1.0 - c) * uncensored)
        else:
            loss = neg_l

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def load_survival_from_csv(label_csv: str, patients: list[str]):
    assert os.path.isfile(label_csv), f"Not found: {label_csv}"
    df = pd.read_csv(label_csv)

    pid_candidates = ["sample", "patient", "patient_id", "case_id", "submitter_id"]
    pid_col = next((c for c in pid_candidates if c in df.columns), None)
    if pid_col is None:
        raise ValueError(f"Cannot find pid col. candidates={pid_candidates}, got={list(df.columns)}")

    time_candidates = ["survival_months", "OS_months", "OS", "time", "t", "months"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"Cannot find time col. candidates={time_candidates}, got={list(df.columns)}")

    censor_candidates = ["censorship", "censored", "censor"]
    censor_col = next((c for c in censor_candidates if c in df.columns), None)

    event_candidates = ["event", "status", "dead"]
    event_col = next((c for c in event_candidates if c in df.columns), None)

    if censor_col is None and event_col is None:
        raise ValueError(f"Need censor or event col. got={list(df.columns)}")

    df[pid_col] = df[pid_col].astype(str).str.strip()
    time_map = dict(zip(df[pid_col], df[time_col]))

    if censor_col is not None:
        censor_map = dict(zip(df[pid_col], df[censor_col]))
        def to_censor(v):
            if pd.isna(v): return None
            return float(v)   # 1=censored, 0=event
    else:
        event_map = dict(zip(df[pid_col], df[event_col]))
        def to_censor(v):
            if pd.isna(v): return None
            ev = int(v)
            return float(1 - ev)
        censor_map = event_map

    t_list, c_list = [], []
    missing = []
    for p in patients:
        tv = time_map.get(p, None)
        cv = censor_map.get(p, None)
        if tv is None or cv is None or pd.isna(tv) or pd.isna(cv):
            t_list.append(float("nan"))
            c_list.append(float("nan"))
            missing.append(p)
        else:
            t_list.append(float(tv))
            c_list.append(float(to_censor(cv)))

    if missing:
        print(f"[WARN] missing survival for {len(missing)} patients. example={missing[:5]}")

    return torch.tensor(t_list, dtype=torch.float32), torch.tensor(c_list, dtype=torch.float32)
