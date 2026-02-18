"""Plotting utilities for training curves and confusion matrices."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(history, plots_dir, run_name):
    """Save loss, accuracy, and F1 curves."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{run_name}: Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / f"{run_name}_loss.png", dpi=300); plt.close()

    # Top-1 accuracy
    plt.figure()
    plt.plot(epochs, history["train_top1"], label="Train")
    plt.plot(epochs, history["val_top1"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Top-1 Accuracy"); plt.title(f"{run_name}: Top-1")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / f"{run_name}_acc.png", dpi=300); plt.close()

    # F1
    plt.figure()
    plt.plot(epochs, history["val_f1_macro"], label="Val F1-macro")
    plt.plot(epochs, history["val_f1_micro"], label="Val F1-micro")
    plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title(f"{run_name}: F1")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / f"{run_name}_f1.png", dpi=300); plt.close()


def plot_confusion_matrix(cm, classes, out_path, normalize=False,
                          title="Confusion Matrix", abbreviations=None):
    """Plot and save a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float32 if normalize else np.int64)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_show = cm / row_sums
        fmt = ".2f"
        text_vals = cm_show
    else:
        cm_show = cm.astype(np.float32)
        text_vals = cm.astype(np.int64)
        fmt = "d"

    tick_labels = [abbreviations.get(c, c) for c in classes] if abbreviations else list(classes)

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.7), max(5, len(classes) * 0.6)))
    im = ax.imshow(cm_show, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(len(tick_labels))
    ax.set(xticks=ticks, yticks=ticks, xticklabels=tick_labels, yticklabels=tick_labels,
           ylabel="True label", xlabel="Predicted label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if len(classes) <= 20:
        thresh = cm_show.max() / 2.0
        for i in range(cm_show.shape[0]):
            for j in range(cm_show.shape[1]):
                ax.text(j, i, format(text_vals[i, j], fmt), ha="center", va="center",
                        color="white" if cm_show[i, j] > thresh else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
