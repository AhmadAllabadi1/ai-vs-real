import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders(
        train_dir='data/train',
        test_dir='data/test',
        batch_size=32,
        val_split=0.2,
):
    torch.manual_seed(42)  # For reproducible split

    # ----- Transforms -----
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    # Base dataset just to get consistent indexing / targets
    base_train = datasets.ImageFolder(train_dir)

    num_train = len(base_train)
    indices = torch.randperm(num_train)
    split = int(num_train * (1 - val_split))

    train_idx = indices[:split]
    val_idx = indices[split:]

    # Apply different transforms for train and val using Subset wrappers
    train_data_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data_full = datasets.ImageFolder(train_dir, transform=eval_transform)

    train_data = Subset(train_data_full, train_idx)
    val_data = Subset(val_data_full, val_idx)

    # Test set (no augmentation)
    test_data = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    classes = base_train.classes
    num_classes = len(classes)

    print("Classes:", classes)
    print("Train size:", len(train_data),
          "| Val size:", len(val_data),
          "| Test size:", len(test_data))

    return train_loader, val_loader, test_loader, num_classes
