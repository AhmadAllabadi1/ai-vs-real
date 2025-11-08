import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import CNN

def main():
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()

    # Init model, loss, optimizer, hyperparameters

    lr = 1e-3
    epochs = 1

    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # --- print every 100 batches ---
            if (batch_idx + 1) % 100 == 0:
                batch_loss = running_loss / total
                batch_acc = correct / total
                print(f"  [Batch {batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f}")
                

        train_loss = running_loss / total
        train_acc = correct / total

        # -------- Validation after each epoch --------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        
    # ----- Evaluate on Test Set -------- 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    test_acc = correct / total if total > 0 else 0
    print(f"Test Acc: {test_acc:.4f}")

    # Save trained weights
    torch.save(model.state_dict(), "model.pth")
    print("Saved model -> model.pth")

if __name__ == "__main__":
    main()

