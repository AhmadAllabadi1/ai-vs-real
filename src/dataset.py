import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders(
        train_dir='data/train',
        test_dir='data/test',
        batch_size=32,
        val_split=0.2,
):
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert images to tensor
        transforms.Normalize(  # Normalize images
            mean=(0.5,0.5,0.5), # RGB images
            std=(0.5,0.5,0.5)
        ),
    ])

    full_train_data = datasets.ImageFolder(train_dir, transform=transform)

    # Split into train and validation sets (80-20 split)
    num_train = len(full_train_data)
    indices = torch.randperm(num_train)
    split = int(num_train * (1 - val_split))

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_data = Subset(full_train_data, train_idx)
    val_data = Subset(full_train_data, val_idx)

    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) #Avoid shuffling for determinsm

    classes = full_train_data.classes
    num_classes = len(classes)

    print("Classes:", classes)
    print("Train size:", len(train_data),
          "| Val size:", len(val_data),
          "| Test size:", len(test_data))

    return train_loader, val_loader, test_loader, num_classes
