# model.py

"""
CNN Architecture

Design Philosophy:
- Start simple (baseline)
- Add capacity if underfitting
- Add regularization if overfitting

CIFAR10 image size: 32x32x3
"""

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # ---------------------------------------
        # Convolutional Layers (Feature Extractor)
        # ---------------------------------------
        # Conv layer extracts spatial patterns

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # MaxPooling reduces spatial size
        self.pool = nn.MaxPool2d(2, 2)

        # ---------------------------------------
        # Fully Connected Layers (Classifier)
        # ---------------------------------------
        # After 3 poolings:
        # 32x32 → 16x16 → 8x8 → 4x4

        self.fc1 = nn.Linear(128 * 4 * 4, 256)

        # Dropout helps reduce overfitting
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        # ----- Feature Extraction -----
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten tensor for fully connected layer
        x = x.view(x.size(0), -1)

        # ----- Classification -----
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x