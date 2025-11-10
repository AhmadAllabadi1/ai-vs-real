import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders(
        train_dir='data/train',
        test_dir='data/test',
        batch_size=64,
        val_split=0.2,
        num_workers=8,
):
    torch.manual_seed(42)  # For reproducible split

    # ----- Transforms -----
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
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

    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    classes = base_train.classes
    num_classes = len(classes)

    print("Classes:", classes)
    print("Train size:", len(train_data),
          "| Val size:", len(val_data),
          "| Test size:", len(test_data))

    return train_loader, val_loader, test_loader, num_classes
