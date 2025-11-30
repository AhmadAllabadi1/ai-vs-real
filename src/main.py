# main.py
import torch
from dataset import get_dataloaders
from baselines.majority import run_majority_baseline
from baselines.logreg import run_logreg_baseline
from train import train_model
from plot import (
    plot_accuracy,
    plot_loss,
    plot_confusion_matrix,
    plot_roc_curves,
)
import time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, num_classes = get_dataloaders()
    """
    start = time.time()
    print("\n=== Baseline 1: Majority ===")
    maj_results = run_majority_baseline(train_loader, val_loader, test_loader)
    end = time.time()
    maj_time = end - start

    start = time.time()
    print("\n=== Baseline 2: Logistic Regression ===")
    logreg_results = run_logreg_baseline(
        train_loader, val_loader, test_loader, num_classes, device, epochs=150, lr=1e-2
    )
    end = time.time()
    log_time = end - start
    """
    start = time.time()
    print("\n=== Main Model: ===")
    model_results = train_model(
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        device,
        epochs=80,
        lr=1e-3,
        save_path="model.pth",
    )
    end = time.time()
    cnn_time = end - start

    # === Plots ===
    plot_accuracy(
        model_results["train_acc_history"],
        model_results["val_acc_history"],
        filename="model_accuracy.png",
    )

    plot_loss(
        model_results["train_loss_history"],
        model_results["val_loss_history"],
        filename="model_loss.png",
    )

    plot_confusion_matrix(
        model_results["confusion_matrix"],
        filename="model_confusion_matrix.png",
    )

    plot_roc_curves(
        model_results["y_true"],
        model_results["y_score"],
        filename="model_roc.png",
    )

    print("\n=== Summary ===")
    #print(f"Majority Baseline:  Val {maj_results['val']:.4f} | Test {maj_results['test']:.4f} | Time {maj_time:.4f} seconds")
    #print(f"LogReg Baseline:    Val {logreg_results['val']:.4f} | Test {logreg_results['test']:.4f} | Time {log_time:.4f} seconds")
    print(f"Model:          Val {model_results['val_acc']:.4f} | Test {model_results['test_acc']:.4f} | Time {cnn_time:.4f} seconds")

if __name__ == "__main__":
    main()
