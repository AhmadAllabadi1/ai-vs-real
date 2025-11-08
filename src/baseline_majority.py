import torch
from collections import Counter
from dataset import get_dataloaders
from torch.utils.data import Subset

def get_majority_class(train_loader):
    train_subset: Subset = train_loader.dataset
    base_dataset = train_subset.dataset          # the original ImageFolder
    indices = train_subset.indices               # indices used for train split

    # collect labels only for train indices
    labels = [base_dataset.targets[i] for i in indices]

    counts = Counter(labels)
    majority_class, freq = counts.most_common(1)[0]
    class_name = base_dataset.classes[majority_class]

    print(f"Majority class: {majority_class} ({class_name}), count={freq}")
    return majority_class, class_name

def eval_majority(loader, majority_class):
    correct = 0
    total = 0
    for _, labels in loader:
        preds = torch.full_like(labels, fill_value=majority_class)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0

def main():
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()

    majority_class, class_name = get_majority_class(train_loader)

    train_acc = eval_majority(train_loader, majority_class)
    val_acc   = eval_majority(val_loader, majority_class)
    test_acc  = eval_majority(test_loader, majority_class)

    print("\n=== Majority Class Baseline ===")
    print(f"Predicting class: {majority_class} ({class_name})")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

if __name__ == "__main__":
    main()
