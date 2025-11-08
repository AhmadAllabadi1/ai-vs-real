from dataset import get_dataloaders

train_loader, val_loader, test_loader, num_classes = get_dataloaders()
images, labels = next(iter(train_loader))

print(f"Number of classes: {num_classes}")
print(f"Batch size: {images.size(0)}")
print("Sample labels:", labels[:10])

# Pick one image tensor and flatten it
img = images[0]  # first image in batch
print("\nSingle image tensor shape:", img.shape)

# Flatten to a vector
img_vector = img.flatten()
print("Flattened vector length:", len(img_vector))

# Print first 50 values
print("\nFirst 50 pixel values:")
print(img_vector[:50])