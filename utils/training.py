"""Training utilities: epoch runner, parameter unfreezing, optimizer builder."""

import contextlib
import torch
import torch.nn as nn
from .metrics import f1_from_confmat, update_confmat, accuracy_topk


# ── Unfreeze helpers ──────────────────────────────────────────────────

def unfreeze_swin3d_tail(model, k=1):
    """Unfreeze the last k transformer blocks + final norm of Swin3D."""
    prefixes = []
    backbone = model

    def _unfreeze_blocks(blocks, base):
        n = len(blocks)
        kk = max(1, min(k, n))
        for i in range(n - kk, n):
            for p in blocks[i].parameters():
                p.requires_grad = True
            prefixes.append(f"{base}.{i}")

    features = getattr(backbone, "features", None)
    if features is not None:
        stages = getattr(features, "stages", None)
        if isinstance(stages, (nn.ModuleList, nn.Sequential)) and len(stages) > 0:
            blks = getattr(stages[-1], "blocks", None)
            if isinstance(blks, (nn.ModuleList, nn.Sequential)) and len(blks) > 0:
                _unfreeze_blocks(blks, f"features.stages.{len(stages)-1}.blocks")

    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad = True
        prefixes.append("norm")

    return prefixes


def unfreeze_r3d_tail(model, k=1):
    """Unfreeze the last k layers of R3D-18."""
    layers = ["layer4", "layer3", "layer2", "layer1"]
    prefixes = []
    for name in layers[:k]:
        layer = getattr(model, name, None)
        if layer is not None:
            for p in layer.parameters():
                p.requires_grad = True
            prefixes.append(name)
    return prefixes


def unfreeze_s3d_tail(backbone, k=1):
    """Unfreeze the last feature blocks of S3D."""
    features = getattr(backbone, "features", None)
    if features is None:
        return []
    children = list(features.children())
    n = len(children)
    kk = max(1, min(k, n))
    prefixes = []
    for i in range(n - kk, n):
        for p in children[i].parameters():
            p.requires_grad = True
        prefixes.append(f"features.{n - kk + i}")
    return prefixes


def unfreeze_mvit_tail(backbone, k=1):
    """Unfreeze the last k blocks of MViTv2."""
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return []
    n = len(blocks)
    kk = max(1, min(k, n))
    prefixes = []
    for i in range(n - kk, n):
        for p in blocks[i].parameters():
            p.requires_grad = True
        prefixes.append(f"blocks.{i}")
    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad = True
        prefixes.append("norm")
    return prefixes


_UNFREEZE_FNS = {
    "swin3d_t": unfreeze_swin3d_tail,
    "r3d_18": unfreeze_r3d_tail,
    "mvit_v2_s": unfreeze_mvit_tail,
    "s3d": unfreeze_s3d_tail,
}


def unfreeze_tail(model, backbone_name, k=1):
    """Unfreeze tail blocks for a given backbone within a fusion model."""
    fn = _UNFREEZE_FNS.get(backbone_name)
    if fn is None:
        return []

    all_prefixes = []
    for attr_name in ["encoder", "rgb_enc", "gaze_enc"]:
        enc = getattr(model, attr_name, None)
        if enc is None:
            continue
        bb = getattr(enc, "backbone", enc)
        prefixes = fn(bb, k)
        all_prefixes.extend([f"{attr_name}.backbone.{p}" for p in prefixes])

    return all_prefixes


def set_frozen_eval(model, tail_prefixes):
    """Keep frozen modules in eval mode (stable BN/Dropout)."""
    trainable = {"head", "tele_enc"} | set(tail_prefixes)
    for name, module in model.named_modules():
        if any(name.startswith(t) for t in trainable):
            continue
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.LayerNorm)):
            module.eval()


# ── Optimizer ─────────────────────────────────────────────────────────

def build_optimizer(model, cfg, tail_prefixes):
    """Build AdamW optimizer with separate LR for head/tele and tail."""
    tcfg = cfg["training"]
    lr_head = tcfg.get("lr_head", 1e-4)
    lr_tail = tcfg.get("lr_tail", 3e-5)
    wd = tcfg.get("weight_decay", 1e-4)

    tail_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and any(n.startswith(pf) for pf in tail_prefixes)]

    head_tele_params = []
    for attr in ["head", "tele_enc"]:
        mod = getattr(model, attr, None)
        if mod is not None:
            head_tele_params.extend(list(mod.parameters()))

    param_groups = [
        {"params": head_tele_params, "lr": lr_head},
        {"params": tail_params, "lr": lr_tail},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=wd)


# ── Epoch runner ──────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, n_classes,
              tail_prefixes, device, train=True, use_amp=True):
    """Run one training or validation epoch. Returns metrics dict."""
    model.train(train)
    set_frozen_eval(model, tail_prefixes)

    total, correct1, correct5, loss_sum = 0, 0, 0, 0.0
    confmat = torch.zeros((n_classes, n_classes), dtype=torch.int64, device="cpu")

    amp_ctx = torch.autocast("cuda", dtype=torch.float16, enabled=use_amp) if use_amp else contextlib.nullcontext()

    for batch in loader:
        *inputs, labels = [t.to(device, non_blocking=True) for t in batch]

        with amp_ctx:
            logits = model(*inputs)
            loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            all_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            scaler.step(optimizer)
            scaler.update()

        bs = labels.size(0)
        total += bs
        preds = logits.argmax(1)
        correct1 += (preds == labels).sum().item()
        correct5 += accuracy_topk(logits, labels, k=5)
        loss_sum += loss.item() * bs
        confmat = update_confmat(confmat, labels.cpu(), preds.cpu(), n_classes)

    macro_f1, micro_f1 = f1_from_confmat(confmat)
    return {
        "loss": loss_sum / max(total, 1),
        "top1": correct1 / max(total, 1),
        "top5": correct5 / max(total, 1),
        "f1_macro": macro_f1,
        "f1_micro": micro_f1,
        "confmat": confmat,
    }
