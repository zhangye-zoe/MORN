# utils/train_one_fold.py
import copy
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

from utils.supcon import SupConLoss
from utils.walk_sampler import TypeConstrainedWalkSampler


# =========================
# Survival: C-index + time-dependent AUC helpers
# =========================
def _surv_risk_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    SurvPath-style risk:
      hazards = sigmoid(logits)
      survival = cumprod(1-hazards)
      risk = -sum(survival)  (higher risk -> worse)
    """
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1)
    return risk


def _to_sksurv_struct(event_time_np: np.ndarray, censorship_np: np.ndarray):
    """
    sksurv structured array:
      event: bool (True if event happened)
      time: float
    censorship: 1=censored, 0=event
    """
    event = (censorship_np == 0)
    y = np.zeros(event.shape[0], dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = event.astype(bool)
    y["time"] = event_time_np.astype(float)
    return y


def _default_auc_times_from_train(
    train_event_time_np: np.ndarray,
    train_censorship_np: np.ndarray,
    num_times: int = 5,
) -> np.ndarray:
    """
    选一组合理的 time grid 给 cumulative_dynamic_auc：
    用 train 中未删失样本的 event time 分位数(10%~90%)。
    """
    event_mask = (train_censorship_np == 0)
    t_event = train_event_time_np[event_mask]
    if t_event.size < 5:
        # fallback：用所有 train time
        t_event = train_event_time_np

    t_event = np.asarray(t_event, dtype=float)
    t_event = t_event[np.isfinite(t_event)]
    if t_event.size == 0:
        return np.array([], dtype=float)

    qs = np.linspace(0.1, 0.9, num_times)
    times = np.quantile(t_event, qs)
    # 去重 + 过滤非正
    times = np.unique(times)
    times = times[times > 0]
    return times.astype(float)


@torch.no_grad()
def _eval_survival_metrics(
    logits: torch.Tensor,
    event_time: torch.Tensor,
    censorship: torch.Tensor,
    idx: torch.Tensor,
    # time-AUC 需要 train 作 reference
    train_event_time: torch.Tensor,
    train_censorship: torch.Tensor,
    train_idx: torch.Tensor,
    auc_times: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    返回:
      cindex
      mean_time_auc  (dynamic AUC over selected time grid; mean of AUCs; never NaN)
    """
    # ---- c-index ----
    risk_np = _surv_risk_from_logits(logits)[idx].detach().cpu().numpy()
    t_np = event_time[idx].detach().cpu().numpy().astype(float)
    c_np = censorship[idx].detach().cpu().numpy().astype(int)

    # 约定：censorship=1 表示删失，0 表示事件
    event_np = (c_np == 0)
    cindex = float(concordance_index_censored(event_np, t_np, risk_np)[0])

    # ---- time-dependent AUC (dynamic AUC) ----
    t_train_np = train_event_time[train_idx].detach().cpu().numpy().astype(float)
    c_train_np = train_censorship[train_idx].detach().cpu().numpy().astype(int)

    y_train = _to_sksurv_struct(t_train_np, c_train_np)
    y_test  = _to_sksurv_struct(t_np, c_np)

    if auc_times is None:
        auc_times = _default_auc_times_from_train(t_train_np, c_train_np, num_times=5)

    # ✅ 默认值：不可计算时不返回 NaN
    mean_auc = 0.5

    # 如果 test 集压根没有事件，dynamic AUC 本来就不可定义 → 直接用默认值
    n_test_events = int(np.sum(event_np))
    if n_test_events == 0:
        return {"cindex": cindex, "mean_time_auc": mean_auc}

    if auc_times is not None and len(auc_times) > 0:
        try:
            times = np.asarray(auc_times, dtype=float)

            # 1) ✅ 裁剪到 sksurv 要求的 test 随访范围: [min(test_time), max(test_time))
            tmin = float(np.min(t_np))
            tmax = float(np.max(t_np))
            times = times[(times >= tmin) & (times < tmax)]
            times = np.unique(times)

            if times.size == 0:
                return {"cindex": cindex, "mean_time_auc": mean_auc}

            # 2) ✅ 再过滤：保证在 time=t 时，test 中至少有 1 个“在 t 之前发生事件”的样本
            #    否则内部会出现 cumsum_tp[-1]==0 -> divide by zero -> auc nan
            valid_times = []
            for t in times:
                if np.sum((t_np <= t) & event_np) > 0:
                    valid_times.append(t)

            if len(valid_times) == 0:
                return {"cindex": cindex, "mean_time_auc": mean_auc}

            valid_times = np.asarray(valid_times, dtype=float)

            # 3) ✅ 计算 AUC，并忽略 nan/inf
            _, aucs = cumulative_dynamic_auc(y_train, y_test, risk_np, valid_times)
            aucs = np.asarray(aucs, dtype=float)
            ok = np.isfinite(aucs)
            if np.any(ok):
                mean_auc = float(np.mean(aucs[ok]))
            # else: 保持默认 mean_auc=0.5

        except Exception:
            # 任意报错都不影响训练，返回默认值
            pass

    return {"cindex": cindex, "mean_time_auc": mean_auc}


# =========================
# Grading: ACC + AUC helpers
# =========================
@torch.no_grad()
def _eval_acc(logits: torch.Tensor, y_true: torch.Tensor, idx: torch.Tensor) -> float:
    pred = torch.argmax(logits[idx], dim=-1)
    acc = (pred == y_true[idx]).float().mean().item()
    return float(acc)


@torch.no_grad()
def _eval_auc_grading(logits: torch.Tensor, y_true: torch.Tensor, idx: torch.Tensor, num_classes: int) -> float:
    """
    支持二分类/多分类：
      - 二分类：用 softmax[:,1] 做 AUC
      - 多分类：macro OVR AUC
    """
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")

    lg = logits[idx].detach().cpu()
    y = y_true[idx].detach().cpu().numpy().astype(int)

    prob = torch.softmax(lg, dim=-1).numpy()

    # 如果某个 split 里类别不全，roc_auc_score 可能会报错 -> 返回 NaN
    # print('y', y)
    # print('prob', prob)
    # print('=' * 100)
    try:
        if num_classes == 2:
            # AUC expects score for positive class
            return float(roc_auc_score(y, prob[:, 1]))
        else:
            # y: (N,), prob: (N, C)

            return float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


# =========================
# Batching + walk aggregation
# =========================
def _iter_batches(idx_cpu: torch.Tensor, bs: int):
    if bs <= 0 or bs >= len(idx_cpu):
        yield idx_cpu
        return
    perm = torch.randperm(len(idx_cpu))
    idx_cpu = idx_cpu[perm]
    for i in range(0, len(idx_cpu), bs):
        yield idx_cpu[i:i + bs]


def _aggregate_walk_embeddings_on_device(
    walks: List[List[Tuple[str, int]]],
    h_dict: Dict[str, torch.Tensor],
    device: torch.device,
    agg: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    walks: list of walk, each walk is [(ntype,id), (ntype,id), ...]
    h_dict: dict[ntype] -> Tensor[num_nodes(ntype), hid_dim] on DEVICE
            must include "patient" and gene_* types
    Return:
        walk_emb: (num_walks, hid_dim)
        start_pids: (num_walks,) LongTensor (DEVICE)  -- the patient id each walk belongs to
    """
    assert agg in ("mean", "sum")
    embs = []
    pids = []

    for w in walks:
        if len(w) == 0:
            continue
        start_t, start_id = w[0]
        pids.append(int(start_id))

        node_vecs = []
        for (t, nid) in w:
            if t not in h_dict:
                continue
            nid_t = torch.tensor([int(nid)], device=device, dtype=torch.long)
            node_vecs.append(h_dict[t].index_select(0, nid_t).squeeze(0))

        if len(node_vecs) == 0:
            any_t = next(iter(h_dict.keys()))
            node_vecs = [torch.zeros(h_dict[any_t].shape[1], device=device, dtype=h_dict[any_t].dtype)]

        node_stack = torch.stack(node_vecs, dim=0)  # (L, H)
        if agg == "mean":
            embs.append(node_stack.mean(dim=0))
        else:
            embs.append(node_stack.sum(dim=0))

    if len(embs) == 0:
        any_t = next(iter(h_dict.keys()))
        H = h_dict[any_t].shape[1]
        return (
            torch.zeros((0, H), device=device, dtype=h_dict[any_t].dtype),
            torch.zeros((0,), device=device, dtype=torch.long),
        )

    walk_emb = torch.stack(embs, dim=0)  # (W, H)
    start_pids = torch.tensor(pids, device=device, dtype=torch.long)
    return walk_emb, start_pids


# =========================
# Main train_one_fold
# =========================
def train_one_fold(
    model,
    G,
    target_ntype: str,
    # --- task switch ---
    task: str = "survival",  # "survival" or "grading"

    # --- survival labels ---
    y_disc: Optional[torch.Tensor] = None,
    event_time: Optional[torch.Tensor] = None,
    censorship: Optional[torch.Tensor] = None,

    # --- grading labels ---
    y_grade: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,

    # --- common ---
    train_idx: torch.Tensor = None,
    val_idx: torch.Tensor = None,
    test_idx: torch.Tensor = None,
    loss_fn=None,
    optimizer=None,
    scheduler=None,
    n_epoch: int = 10,
    eval_every: int = 1,
    clip: float = 1.0,

    # --- CL (walk supcon) ---
    lambda_walk_cl: float = 0.1,
    cl_temperature: float = 0.1,
    walk_len: int = 3,
    walks_per_patient: int = 4,

    # biological constraints
    allowed_next_types: Optional[Dict[str, list]] = None,
    metapath_types_list: Optional[list] = None,
    edge_weight_key: str = "w",

    # --- stage-wise training ---
    warmup_epochs: int = 10,

    # --- batching (optional) ---
    batch_size: int = 8,  # 0 means full-batch over train_idx for CL

    # --- survival time-AUC config ---
    time_auc_times: Optional[List[float]] = None,  # 如果你想固定时间点，例如 [365, 730, 1095]
) -> Dict[str, Any]:

    task = str(task).lower()
    assert task in ("survival", "grading"), f"task must be 'survival'|'grading', got {task}"

    if task == "survival":
        assert y_disc is not None and event_time is not None and censorship is not None, \
            "survival task requires y_disc, event_time, censorship"
        device = y_disc.device
    else:
        assert y_grade is not None and num_classes is not None, \
            "grading task requires y_grade and num_classes"
        device = y_grade.device

    best_val = -1.0
    best_state = None
    best_epoch = -1
    best_test_tracked = float("nan")

    supcon = SupConLoss(temperature=cl_temperature).to(device)

    # keep a CPU graph for walk sampling (avoid moving every time)
    G_cpu = G.to("cpu")

    sampler = TypeConstrainedWalkSampler(
        G=G_cpu,
        allowed_next_types=allowed_next_types if (allowed_next_types and len(allowed_next_types) > 0) else None,
        metapath_types_list=metapath_types_list,
        edge_dir="out",
        # 如果你想按边权采样：prob_key=edge_weight_key
        prob_key=None,
        seed=0,
    )

    train_idx_cpu = train_idx.detach().cpu()

    # parse custom time-AUC times
    auc_times_np = [24, 36, 48, 60] #None
    if time_auc_times is not None and len(time_auc_times) > 0:
        auc_times_np = np.array(time_auc_times, dtype=float)

    for epoch in range(1, n_epoch + 1):
        model.train()

        # stage-wise
        if epoch <= warmup_epochs:
            model.set_phase("omics")
        else:
            model.set_phase("full")

        optimizer.zero_grad(set_to_none=True)

        # ---- single forward per epoch for both main loss + CL ----
        # require model returns: logits, omics_h(dict), patient_embed
        out = model(G, target_ntype, return_omics_h=True, return_patient_embed=True)
        logits = out[0]
        omics_h = out[-2]        # dict[gene_*] -> (N,H)
        patient_embed = out[-1]  # (Np,H)

        # -------------------------
        # main supervised loss
        # -------------------------
        if task == "survival":
            loss_main = loss_fn(
                logits[train_idx],
                y_disc[train_idx],
                event_time[train_idx],
                censorship[train_idx],
            )
        else:
            loss_main = loss_fn(logits[train_idx], y_grade[train_idx])

        loss = loss_main

        # -------------------------
        # walk supcon
        # -------------------------
        loss_cl_total = torch.tensor(0.0, device=device)
        if lambda_walk_cl > 0:
            h_dict = dict(omics_h)
            h_dict["patient"] = patient_embed

            nb = 0
            for b_idx_cpu in _iter_batches(train_idx_cpu, batch_size):
                nb += 1
                walks = sampler.sample_walk_node_ids(
                    start_ntype="patient",
                    start_node_ids=b_idx_cpu,
                    walk_len=walk_len,
                    walks_per_node=walks_per_patient,
                )

                walk_emb, start_pids = _aggregate_walk_embeddings_on_device(
                    walks=walks,
                    h_dict=h_dict,
                    device=device,
                    agg="mean",
                )

                if walk_emb.shape[0] == 0:
                    continue

                if task == "survival":
                    labels = y_disc[start_pids].long()
                else:
                    labels = y_grade[start_pids].long()

                loss_cl_total = loss_cl_total + supcon(walk_emb, labels)

            if nb > 0:
                loss_cl_total = loss_cl_total / float(nb)

            loss = loss + lambda_walk_cl * loss_cl_total

        # backward
        loss.backward()
        if clip is not None and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # -------------------------
        # eval
        # -------------------------
        if (epoch % eval_every) == 0:
            model.eval()
            with torch.no_grad():
                logits_eval = model(G, target_ntype)

            # print('auc time np', auc_times_np)
            # print('=' * 100)

            if task == "survival":
                # print('train idx', train_idx)
                # print('val idx', val_idx)
                # print('test idx', test_idx)
                # print('=' * 100)
                train_m = _eval_survival_metrics(
                    logits_eval, event_time, censorship, train_idx,
                    train_event_time=event_time, train_censorship=censorship, train_idx=train_idx,
                    auc_times=auc_times_np,
                )
                val_m = _eval_survival_metrics(
                    logits_eval, event_time, censorship, val_idx,
                    train_event_time=event_time, train_censorship=censorship, train_idx=train_idx,
                    auc_times=auc_times_np,
                )
                test_m = _eval_survival_metrics(
                    logits_eval, event_time, censorship, test_idx,
                    train_event_time=event_time, train_censorship=censorship, train_idx=train_idx,
                    auc_times=auc_times_np,
                )
                # print('train m', train_m)
                # print('val m', val_m)
                # print('test m', test_m) 
                # print('=' * 100)

                # model selection: still by val c-index (你也可以改成 val mean_time_auc)
                if val_m["cindex"] > best_val:
                    best_val = val_m["cindex"]
                    best_epoch = epoch
                    best_test_tracked = test_m["cindex"]
                    best_state = copy.deepcopy(model.state_dict())

                print(
                    f"[E{epoch:03d}] task=survival phase={getattr(model, '_phase', 'NA')} "
                    f"loss={float(loss.item()):.4f} main={float(loss_main.item()):.4f} cl={float(loss_cl_total.item()):.4f} | "
                    f"c-index train/val/test = {train_m['cindex']:.4f}/{val_m['cindex']:.4f}/{test_m['cindex']:.4f} | "
                    f"time-AUC(mean) train/val/test = {train_m['mean_time_auc']:.4f}/{val_m['mean_time_auc']:.4f}/{test_m['mean_time_auc']:.4f}"
                )

            else:
                train_acc = _eval_acc(logits_eval, y_grade, train_idx)
                val_acc = _eval_acc(logits_eval, y_grade, val_idx)
                test_acc = _eval_acc(logits_eval, y_grade, test_idx)

                train_auc = _eval_auc_grading(logits_eval, y_grade, train_idx, num_classes=num_classes)
                val_auc = _eval_auc_grading(logits_eval, y_grade, val_idx, num_classes=num_classes)
                test_auc = _eval_auc_grading(logits_eval, y_grade, test_idx, num_classes=num_classes)

                # model selection: by val ACC (你也可以改成 val_auc)
                if val_acc > best_val:
                    best_val = val_acc
                    best_epoch = epoch
                    best_test_tracked = test_acc
                    best_state = copy.deepcopy(model.state_dict())

                print(
                    f"[E{epoch:03d}] task=grading phase={getattr(model, '_phase', 'NA')} "
                    f"loss={float(loss.item()):.4f} main={float(loss_main.item()):.4f} cl={float(loss_cl_total.item()):.4f} | "
                    f"ACC train/val/test = {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f} | "
                    f"AUC train/val/test = {train_auc:.4f}/{val_auc:.4f}/{test_auc:.4f}"
                )

    # final
    final_test = float("nan")
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        model.eval()
        with torch.no_grad():
            logits_best = model(G, target_ntype)

        if task == "survival":
            m = _eval_survival_metrics(
                logits_best, event_time, censorship, test_idx,
                train_event_time=event_time, train_censorship=censorship, train_idx=train_idx,
                auc_times=auc_times_np,
            )
            final_test = m["cindex"]
            final_test_time_auc = m["mean_time_auc"]
        else:
            final_test = _eval_acc(logits_best, y_grade, test_idx)
            final_test_auc = _eval_auc_grading(logits_best, y_grade, test_idx, num_classes=num_classes)

    ret = {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_test_tracked": best_test_tracked,
        "final_test": final_test,
        "best_state": best_state,
        "task": task,
    }

    if task == "survival":
        ret["final_test_time_auc"] = locals().get("final_test_time_auc", float("nan"))
    else:
        ret["final_test_auc"] = locals().get("final_test_auc", float("nan"))

    return ret