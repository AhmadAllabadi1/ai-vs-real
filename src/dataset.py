from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(
        train_dir='data/train',
        test_dir='data/test',
        batch_size=32,
):
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert images to tensor
        transforms.Normalize(  # Normalize images
            mean=(0.5,0.5,0.5), # RGB images
            std=(0.5,0.5,0.5)
        ),
    ])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) #Avoid shuffling for determinsm
    print("Classes: ", train_data.classes)
    return train_loader, test_loader, len(train_data.classes)
