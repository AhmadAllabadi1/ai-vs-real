# main.py
import torch
from dataset import get_dataloaders
from baselines.majority import run_majority_baseline
from baselines.logreg import run_logreg_baseline
from train import train_cnn
import time

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
        epochs=150,
        lr=1e-3,
        save_path="model_cnn.pth",
    )
    end = time.time()
    cnn_time = end - start

    print("\n=== Summary ===")
    '''
    print(f"Majority Baseline:  Val {maj_results['val']:.4f} | Test {maj_results['test']:.4f} | Time {maj_time:.4f} seconds")
    print(f"LogReg Baseline:    Val {logreg_results['val']:.4f} | Test {logreg_results['test']:.4f}  | Time {log_time:.4f} seconds")
    '''
    print(f"CNN Model:          Val {cnn_results['val_acc']:.4f} | Test {cnn_results['test_acc']:.4f}  | Time {cnn_time:.4f} seconds")

if __name__ == "__main__":
    main()
