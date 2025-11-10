# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from metrics import eval_with_metrics
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler

def train_cnn(train_loader, val_loader, test_loader, num_classes,
              device, epochs=5, lr=1e-3, weight_decay=1e-4, save_path="model_cnn.pth"):

    model = CNN(num_classes=num_classes).to(device)
    torch.backends.cudnn.benchmark = True
    #model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    best_val_acc = 0.0
    scaler = GradScaler('cuda')

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

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

        # ---- Validate ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs = model(images)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        scheduler.step()

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), save_path)

    # ---- Test with best model ----
    model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_to_load.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

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
