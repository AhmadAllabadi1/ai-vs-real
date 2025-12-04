import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from metrics import eval_with_metrics
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler


def train_cnn(train_loader, val_loader, test_loader, num_classes,
              device, epochs=5, lr=3e-4, weight_decay=1e-4,
              save_path="model_cnn.pth"):

    model = CNN(num_classes=num_classes).to(device)
    torch.backends.cudnn.benchmark = True

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    scaler = GradScaler('cuda')
    best_val_acc = 0.0

    for epoch in range(epochs):

        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        model.eval()
        correct, total, val_running_loss = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_running_loss / total
        val_acc = correct / total

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        scheduler.step()

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss {train_loss:.4f} | "
              f"Train Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} | "
              f"Val Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(model_to_save.state_dict(), save_path)

    model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_to_load.load_state_dict(torch.load(save_path, map_location=device))

    test_acc, test_prec, test_rec, test_f1, test_cm, all_labels, all_probs = \
        eval_with_metrics(model, test_loader, device)

    print("\n=== CNN Test Metrics ===")
    print(f"Test Acc:       {test_acc:.4f}")
    print(f"Test Precision:{test_prec:.4f}")
    print(f"Test Recall:   {test_rec:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)

    return {
        "val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_prec": float(test_prec),
        "test_rec": float(test_rec),
        "test_f1": float(test_f1),

        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,

        "y_true": all_labels,
        "y_score": all_probs,
        "confusion_matrix": test_cm,
    }
