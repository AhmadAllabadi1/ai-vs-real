# plot.py
import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)


PLOTS_DIR = "plots"


def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_accuracy(
    train_acc: List[float],
    val_acc: List[float],
    filename: str = "accuracy.png",
):
    """
    Plot Train / Val accuracy vs. epoch and save to plots/.
    """
    _ensure_plots_dir()
    epochs = range(1, len(train_acc) + 1)

    plt.figure()
    plt.plot(epochs, train_acc, marker="o", label="Train Acc")
    plt.plot(epochs, val_acc, marker="o", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_loss(
    train_loss: List[float],
    val_loss: List[float],
    filename: str = "loss.png",
):
    """
    Plot Train / Val loss vs. epoch and save to plots/.
    """
    _ensure_plots_dir()
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    filename: str = "confusion_matrix.png",
):
    """
    Plot confusion matrix and save to plots/.
    Expects cm as (num_classes, num_classes).
    """
    _ensure_plots_dir()

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1e-12)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format=".2f" if normalize else "d", colorbar=True)
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
    filename: str = "roc.png",
):
    """
    Plot ROC curve(s) and save to plots/.

    Args:
        y_true: shape (N,), integer class labels [0..C-1] or {0,1} for binary.
        y_score: shape (N, C) for multi-class (probabilities or logits),
                 or shape (N,) / (N,1) for binary (score for positive class).
    """
    _ensure_plots_dir()
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Binary case
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        # Assume positive class is 1
        pos_score = y_score.ravel()
        fpr, tpr, _ = roc_curve(y_true, pos_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Binary)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

    else:
        # Multi-class: one-vs-rest
        n_classes = y_score.shape[1]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        # One-hot encode y_true
        y_true_oh = np.zeros_like(y_score)
        y_true_oh[np.arange(y_true.shape[0]), y_true] = 1

        plt.figure()
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.4f})")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (One-vs-Rest)")
        plt.legend(fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
