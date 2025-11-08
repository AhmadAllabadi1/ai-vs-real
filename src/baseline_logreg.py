import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders

class LogisticRegressionModel(nn.Module):
    """Single-layer linear classifier (no convolutions)."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten images
        return self.fc(x)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    # Load data
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get input dimension (flattened image size)
    sample_imgs, _ = next(iter(train_loader))
    input_dim = sample_imgs.view(sample_imgs.size(0), -1).shape[1]
    print(f"Input dimension: {input_dim}, Num classes: {num_classes}")

    # Initialize model, loss, optimizer
    model = LogisticRegressionModel(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train for a few epochs (no need for long training)
    epochs = 3
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
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate
    train_acc = evaluate(model, train_loader, device)
    val_acc = evaluate(model, val_loader, device)
    test_acc = evaluate(model, test_loader, device)

    print("\n=== Logistic Regression Baseline ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

if __name__ == "__main__":
    main()
