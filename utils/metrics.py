"""Evaluation metrics: F1, confusion matrix, top-k accuracy."""

import torch


@torch.no_grad()
def f1_from_confmat(confmat):
    """Compute macro and micro F1 from a confusion matrix."""
    tp = confmat.diag().float()
    fp = confmat.sum(0).float() - tp
    fn = confmat.sum(1).float() - tp

    denom = (2 * tp + fp + fn).clamp(min=1e-8)
    per_class_f1 = (2 * tp) / denom
    macro_f1 = per_class_f1.mean().item()

    tp_sum, fp_sum, fn_sum = tp.sum(), fp.sum(), fn.sum()
    micro_denom = (2 * tp_sum + fp_sum + fn_sum).clamp(min=1e-8)
    micro_f1 = (2 * tp_sum / micro_denom).item()

    return macro_f1, micro_f1


@torch.no_grad()
def update_confmat(confmat, y_true, y_pred, n_classes):
    """Accumulate predictions into an existing confusion matrix."""
    k = (y_true * n_classes + y_pred).to(torch.int64)
    confmat += torch.bincount(k, minlength=n_classes ** 2).view(n_classes, n_classes)
    return confmat


def accuracy_topk(logits, targets, k):
    """Compute top-k accuracy (returns count of correct predictions)."""
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1, largest=True, sorted=True).indices
    return (topk == targets.unsqueeze(1)).any(dim=1).sum().item()
