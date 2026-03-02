# data.py

"""
This file is responsible for:

1) Downloading CIFAR-10 dataset
2) Applying data augmentation
3) Creating DataLoaders

DataLoader is critical because:
- It batches the data
- It shuffles training data
- It makes GPU training efficient
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config


def get_dataloaders():

    # ==========================================
    # Training Transformations (Data Augmentation)
    # ==========================================
    # Why augmentation?
    # It artificially increases dataset diversity.
    # This reduces overfitting (variance problem).

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),      # Random mirror
        transforms.RandomCrop(32, padding=4),   # Random crop with padding
        transforms.ToTensor()
    ])

    # ==========================================
    # Test Transformations
    # ==========================================
    # We DO NOT augment test data.
    # We want real evaluation.
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load training dataset
    train_dataset = datasets.CIFAR10(
        root=Config.DATA_PATH,
        train=True,
        transform=transform_train,
        download=True
    )

    # Load test dataset
    test_dataset = datasets.CIFAR10(
        root=Config.DATA_PATH,
        train=False,
        transform=transform_test,
        download=True
    )

    # DataLoader handles batching + shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True  # IMPORTANT for training
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    print("[INFO] CIFAR-10 dataset loaded successfully")

    return train_loader, test_loader