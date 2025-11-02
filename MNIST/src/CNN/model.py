import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Conv Layer 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #downsample by 2 

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Conv Layer 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #downsample by 2 

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv Layer 3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),  # Flattened input to FC (64 output channels, 3x3 feature map size) to 128 neurons
            nn.ReLU(),
            nn.Dropout(0.5),  # Prevent overfitting
            nn.Linear(128, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
