import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1) # First convolution layer
        self.pool = nn.pool = nn.MaxPool2d(2,2) # Pool layer
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1) # Second convolution layer
        self.fc1 = nn.Linear(32 * 56 * 56, num_classes)
    
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x))) # Conv1 + ReLU + Pool
        x = self.pool(nn.ReLU()(self.conv2(x))) # Conv2 + ReLU + Pool
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x) # Fully connected layer
        return x