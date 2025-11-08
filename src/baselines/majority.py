# baselines/majority.py

import torch
from collections import Counter
from torch.utils.data import Subset

def get_majority_class(train_loader):
    ds = train_loader.dataset

    # If train_loader comes from a random split, it's likely a Subset
    if isinstance(ds, Subset):
        base_dataset = ds.dataset
        indices = ds.indices
        # ImageFolder-style dataset: labels usually in .targets
        labels = [base_dataset.targets[i] for i in indices]
    else:
        # Fallback: just iterate (slower, but robust)
        labels = []
        for _, y in train_loader:
            labels.extend(y.tolist())
        base_dataset = ds

    counts = Counter(labels)
    majority_class, freq = counts.most_common(1)[0]

    # Try to get human-readable class name if available
    if hasattr(base_dataset, "classes"):
        class_name = base_dataset.classes[majority_class]
    else:
        class_name = str(majority_class)

    print(f"Majority class: {majority_class} ({class_name}), count={freq}")
    return majority_class, class_name

def eval_majority(loader, majority_class):
    correct = 0
    total = 0
    for _, labels in loader:
        # predict the majority class for every sample
        preds = torch.full_like(labels, fill_value=majority_class)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0

def run_majority_baseline(train_loader, val_loader, test_loader):
    majority_class, class_name = get_majority_class(train_loader)

    train_acc = eval_majority(train_loader, majority_class)
    val_acc   = eval_majority(val_loader, majority_class)
    test_acc  = eval_majority(test_loader, majority_class)

    print("\n=== Majority Class Baseline ===")
    print(f"Predicting class: {majority_class} ({class_name})")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    # match what main.py expects
    return {"val": val_acc, "test": test_acc}
