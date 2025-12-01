# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNNViTHybrid
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model(
    train_loader,
    val_loader,
    test_loader,
    num_classes,
    device,
    epochs=5,
    lr=1e-3,
    weight_decay=1e-4,
    save_path="model.pth",
):
    model = CNNViTHybrid(num_classes=num_classes).to(device)
    torch.backends.cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    """
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    """
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0
    scaler = GradScaler('cuda')

    # histories for plotting
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total if total > 0 else 0.0

        # ---- Validate ----
        model.eval()
        correct, total, val_running_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        val_loss = val_running_loss / total if total > 0 else 0.0

        # record histories
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        scheduler.step()

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}"
        )

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # ---- Test with best model ----
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    preds = all_probs.argmax(axis=1)

    test_acc = (preds == all_labels).mean()
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="weighted", zero_division=0
    )
    test_cm = confusion_matrix(all_labels, preds)

    print("\n=== Model Test Metrics ===")
    print(f"Test Acc:      {test_acc:.4f}")
    print(f"Test Precision:{test_prec:.4f}")
    print(f"Test Recall:   {test_rec:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)

    # keep old keys for summary + add everything needed for plots
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
