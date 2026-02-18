"""Focal Cross-Entropy loss for class-imbalanced classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCrossEntropy(nn.Module):
    """Focal loss: down-weights well-classified examples.

    Args:
        gamma: focusing parameter (default 2.0)
        alpha: per-class weights tensor (optional)
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            a = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", a if a.ndim else a.unsqueeze(0))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp(min=1e-8)
        loss = ((1 - pt) ** self.gamma) * (-log_pt)

        if self.alpha is not None:
            loss = self.alpha.gather(0, targets) * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
