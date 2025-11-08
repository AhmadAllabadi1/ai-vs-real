import torch
import torch.nn as nn
import torch.optim as optim
from metrics import eval_with_metrics  

class LogisticRegressionModel(nn.Module):
    """Single-layer linear classifier (no convolutions)."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten images
        return self.fc(x)


def run_logreg_baseline(train_loader, val_loader, test_loader, num_classes, device,
                        epochs=3, lr=0.01):
    # Infer input dimension from one training batch
    sample_imgs, _ = next(iter(train_loader))
    input_dim = sample_imgs.view(sample_imgs.size(0), -1).shape[1]

    print(f"[LogReg] Input dim: {input_dim}, Num classes: {num_classes}")

    model = LogisticRegressionModel(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # ---- Train ----
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"[LogReg][Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # ---- Evaluate using metrics ----
    print("\n=== Logistic Regression Baseline Metrics ===")
    val_acc, val_prec, val_rec, val_f1, val_cm = eval_with_metrics(model, val_loader, device)
    test_acc, test_prec, test_rec, test_f1, test_cm = eval_with_metrics(model, test_loader, device)

    print(f"Val Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
    print(f"Test Acc: {test_acc:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | F1: {test_f1:.4f}")
    print("Test Confusion Matrix:\n", test_cm)

    # match what main.py expects
    return {"val": val_acc, "test": test_acc}
