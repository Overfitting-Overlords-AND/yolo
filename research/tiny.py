import torch
import torch.nn as nn

class TinyYOLO(nn.Module):
    def __init__(self, num_classes):
        super(TinyYOLO, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Example conv layer
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            # Add more layers as per YOLO architecture
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 500),  # Example linear layer
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(500, num_classes * 5)  # num_classes * 5 for class, x, y, w, h
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Example usage
num_classes = 20  # Example for 20 classes
model = TinyYOLO(num_classes)

# Example input tensor
input_tensor = torch.rand(1, 3, 448, 448)  # Example image size for YOLO is 448x448

# Forward pass
output = model(input_tensor)
print(output.shape)
