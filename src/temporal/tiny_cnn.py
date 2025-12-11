import torch.nn as nn


# A smaller CNN model for quick inference block classification (fire/smoke vs None)
class TinyCNN(nn.Module):
    FIRE_SMOKE_CLASS_NAMES = ['fire_smoke', 'none']
    FIRE_SMOKE_CLASS_IDX = 0  # index of fire/smoke class
    def __init__(self, num_classes=2):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),  # Add BatchNorm for stabilization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Add BatchNorm for stabilization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 32),  # Reduce units to lower model capacity
            nn.ReLU(),
            nn.Dropout(0.4),  # Increase dropout rate for stronger regularization
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
