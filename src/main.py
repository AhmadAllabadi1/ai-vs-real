# main.py
import torch
from dataset import get_dataloaders
from baselines.majority import run_majority_baseline
from baselines.logreg import run_logreg_baseline
from train import train_cnn
import time
from plot import (
    plot_accuracy,
    plot_loss,
    plot_confusion_matrix,
    plot_roc_curves,
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, num_classes = get_dataloaders()
    '''
    start = time.time()
    print("\n=== Baseline 1: Majority ===")
    maj_results = run_majority_baseline(train_loader, val_loader, test_loader)
    end = time.time()
    maj_time = end - start

    start = time.time()
    print("\n=== Baseline 2: Logistic Regression ===")
    logreg_results = run_logreg_baseline(
        train_loader, val_loader, test_loader, num_classes, device
    )
    end = time.time()
    log_time = end - start
    '''

    start = time.time()
    print("\n=== Main Model: CNN ===")
    cnn_results = train_cnn(
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        device,
        epochs=120,
        lr=3e-4,
        save_path="model_cnn.pth",
    )
    end = time.time()
    cnn_time = end - start

    plot_accuracy(
        cnn_results["train_acc_history"],
        cnn_results["val_acc_history"],
        filename="cnn8model_accuracy.png",
    )

    plot_loss(
        cnn_results["train_loss_history"],
        cnn_results["val_loss_history"],
        filename="cnn8model_loss.png",
    )

    plot_confusion_matrix(
        cnn_results["confusion_matrix"],
        filename="cnn8model_confusion_matrix.png",
    )

    plot_roc_curves(
        cnn_results["y_true"],
        cnn_results["y_score"],
        filename="cnn8model_roc.png",
    )

    print("\n=== Summary ===")
    '''
    print(f"Majority Baseline:  Val {maj_results['val']:.4f} | Test {maj_results['test']:.4f} | Time {maj_time:.4f} seconds")
    print(f"LogReg Baseline:    Val {logreg_results['val']:.4f} | Test {logreg_results['test']:.4f}  | Time {log_time:.4f} seconds")
    '''
    print(f"CNN Model:          Val {cnn_results['val_acc']:.4f} | Test {cnn_results['test_acc']:.4f}  | Time {cnn_time:.4f} seconds")

if __name__ == "__main__":
    main()
