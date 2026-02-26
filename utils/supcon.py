# utils/supcon.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for a batch of embeddings.
    - features: (B, D), already L2-normalized or not (we normalize inside)
    - labels: (B,)
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Implements supervised contrastive loss (Khosla et al. 2020 style).
        """
        assert features.dim() == 2, f"features must be (B,D), got {features.shape}"
        assert labels.dim() == 1, f"labels must be (B,), got {labels.shape}"

        device = features.device
        B = features.size(0)

        feats = F.normalize(features, dim=1)
        logits = feats @ feats.t() / self.temperature  # (B,B)

        # mask out self
        logits_mask = torch.ones((B, B), device=device, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).to(device=device) & logits_mask  # positives exclude self

        # for numerical stability
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        exp_logits = torch.exp(logits) * logits_mask  # remove self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        # mean over positives
        pos_count = pos_mask.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count

        loss = -mean_log_prob_pos.mean()
        return loss
