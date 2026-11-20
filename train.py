#!/usr/bin/env python3
"""Unified training script for MOTOR rider behavior / legality classification.

Usage:
    python train.py --config configs/swin3d_gaze_tele.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from models.fusion import build_model
from datasets.clip_dataset import ClipDataset
from datasets.rgb_gaze_dataset import RGBGazeDataset
from datasets.rgb_tele_dataset import RGBTeleDataset
from datasets.rgb_gaze_tele_dataset import RGBGazeTeleDataset
from utils.losses import FocalCrossEntropy
from utils.training import run_epoch, build_optimizer, unfreeze_tail
from utils.plotting import plot_training_curves, plot_confusion_matrix
from utils.labels import BEHAVIOR_CLASSES, BEHAVIOR_ABBREV


# ── Dataset builder ───────────────────────────────────────────────────

def build_dataset(df, cfg, mode):
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    fusion = mcfg["fusion"]
    backbone = mcfg.get("backbone", "swin3d_t")
    clip_len = dcfg.get("frames_per_clip", 16)
    img_size = dcfg.get("img_size", 224)
    label_col = dcfg.get("label_col", "label")
    path_col = dcfg.get("path_col", "path_video")

    common = dict(df=df, clip_len=clip_len, backbone=backbone, mode=mode,
                  img_size=img_size, label_col=label_col, path_col=path_col)

    if fusion in ("single", "cross_attn"):
        return ClipDataset(**common)
    elif fusion == "two_stream":
        return RGBGazeDataset(**common, gaze_col=dcfg.get("gaze_col", "path_gaze"))
    elif fusion == "rgb_tele":
        return RGBTeleDataset(**common, tele_len=mcfg.get("tele_len", 128))
    elif fusion == "three_stream":
        return RGBGazeTeleDataset(**common, gaze_col=dcfg.get("gaze_col", "path_gaze"),
                                  tele_len=mcfg.get("tele_len", 128))
    else:
        raise ValueError(f"Unknown fusion: {fusion}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MOTOR unified training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Seed
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Load manifest
    dcfg = cfg["data"]
    manifest = dcfg["manifest_csv"]
    split_col = dcfg.get("split_col", "split")
    label_col = dcfg.get("label_col", "label")

    df = pd.read_csv(manifest)
    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col] == "val"].copy()
    n_classes = int(df[label_col].astype(int).max()) + 1
    cfg["data"]["n_classes"] = n_classes

    print(f"{len(train_df)} train | {len(val_df)} val | {n_classes} classes")

    # Datasets and loaders
    tcfg = cfg["training"]
    batch_size = tcfg.get("batch_size", 8)
    num_workers = dcfg.get("num_workers", 4)

    train_ds = build_dataset(train_df, cfg, "train")
    val_ds = build_dataset(val_df, cfg, "val")
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_amp)

    # Model
    model = build_model(cfg).to(device)
    backbone = cfg["model"].get("backbone", "swin3d_t")
    tail_k = tcfg.get("tail_blocks_k", 1)
    tail_prefixes = unfreeze_tail(model, backbone, tail_k)

    # Ensure head and tele_enc are trainable
    for attr in ["head", "tele_enc"]:
        mod = getattr(model, attr, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,} total | {trainable:,} trainable")
    if tail_prefixes:
        print(f"Unfrozen tails: {tail_prefixes}")

    # Loss
    train_labels = torch.tensor(train_df[label_col].astype(int).tolist())
    counts = torch.bincount(train_labels, minlength=n_classes).float()
    alpha = counts.sum() / (n_classes * counts.clamp(min=1))
    gamma = tcfg.get("focal_gamma", 2.0)
    criterion = FocalCrossEntropy(gamma=gamma, alpha=alpha).to(device)

    # Optimizer and scheduler
    optimizer = build_optimizer(model, cfg, tail_prefixes)
    epochs = tcfg.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Output paths
    ocfg = cfg.get("output", {})
    plots_dir = Path(ocfg.get("plots_dir", "outputs/plots"))
    ckpt_dir = Path(ocfg.get("checkpoints_dir", "outputs/checkpoints"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{backbone}_{cfg['model']['fusion']}"
    if dcfg.get("task") == "legality":
        run_name += "_legal"

    # Training loop
    history = {k: [] for k in ["train_loss", "val_loss", "train_top1", "val_top1",
                                "train_top5", "val_top5", "val_f1_macro", "val_f1_micro"]}
    best_top1 = 0.0
    best_cm = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr = run_epoch(model, train_loader, criterion, optimizer, scaler,
                       n_classes, tail_prefixes, device, train=True, use_amp=use_amp)
        va = run_epoch(model, val_loader, criterion, optimizer, scaler,
                       n_classes, tail_prefixes, device, train=False, use_amp=use_amp)
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_top1"].append(tr["top1"])
        history["val_top1"].append(va["top1"])
        history["train_top5"].append(tr["top5"])
        history["val_top5"].append(va["top5"])
        history["val_f1_macro"].append(va["f1_macro"])
        history["val_f1_micro"].append(va["f1_micro"])

        if va["top1"] > best_top1:
            best_top1 = va["top1"]
            best_cm = va["confmat"].clone()
            torch.save(model.state_dict(), ckpt_dir / f"best_{run_name}.pt")

        elapsed = time.time() - t0
        print(f"E{epoch:02d} | "
              f"Train T1 {tr['top1']*100:5.1f}% F1m {tr['f1_macro']*100:5.1f}% | "
              f"Val T1 {va['top1']*100:5.1f}% F1m {va['f1_macro']*100:5.1f}% | "
              f"Loss {tr['loss']:.3f}/{va['loss']:.3f} | "
              f"Best {best_top1*100:.1f}% | {elapsed:.1f}s")

    # Save final model
    torch.save(model.state_dict(), ckpt_dir / f"final_{run_name}.pt")

    # Plots
    plot_training_curves(history, plots_dir, run_name)

    # Confusion matrices
    task = dcfg.get("task", "behavior")
    if task == "behavior" and n_classes <= len(BEHAVIOR_CLASSES):
        classes = BEHAVIOR_CLASSES[:n_classes]
        abbrev = BEHAVIOR_ABBREV
    else:
        classes = [str(i) for i in range(n_classes)]
        abbrev = None

    if best_cm is not None:
        cm_np = best_cm.cpu().numpy()
        plot_confusion_matrix(cm_np, classes,
                              plots_dir / f"{run_name}_cm.png",
                              title=f"{run_name}: Confusion Matrix",
                              abbreviations=abbrev)
        plot_confusion_matrix(cm_np, classes,
                              plots_dir / f"{run_name}_cm_norm.png",
                              normalize=True,
                              title=f"{run_name}: Normalized Confusion Matrix",
                              abbreviations=abbrev)

    print(f"\nBest Top-1: {best_top1*100:.2f}%")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
