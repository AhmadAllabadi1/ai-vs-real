# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from metrics import eval_with_metrics

def train_cnn(train_loader, val_loader, test_loader, num_classes,
              device, epochs=5, lr=1e-3, save_path="model_cnn.pth"):

    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validate ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # ---- Test with best model ----
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_acc, test_prec, test_rec, test_f1, test_cm = eval_with_metrics(
        model, test_loader, device
    )

    print("\n=== CNN Test Metrics ===")
    print(f"Test Acc:      {test_acc:.4f}")
    print(f"Test Precision:{test_prec:.4f}")
    print(f"Test Recall:   {test_rec:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)

    return {
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_prec": test_prec,
        "test_rec": test_rec,
        "test_f1": test_f1,
    }
