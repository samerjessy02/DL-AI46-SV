# main.py

"""
Main entry point of the experiment.

This file:
1) Sets reproducibility
2) Loads data
3) Builds model
4) Runs sanity check
5) Trains model
6) Evaluates performance
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from utils import set_seed
from model import CNN
from train import train_epoch, evaluate, train_single_sample
from data import get_dataloaders

#Golden Rule

# 1) Sanity Check:
#    Train on a single sample to verify the pipeline works ----> We must overfit on a single datapoint and ensure accuracy is 100% 
#This will guarantee that the model is working perfectlt and there's nothing wrong with the logic.

# 2) Establish Baseline:
#    Start with a simple CNN and observe learning behavior. We do this by havinga  simple model
# Because if we add complexity too soon
# we won’t know what caused improvement or failure.

# 3) Reduce Bias:
#    If the model underfits (low training accuracy),
#    increase model capacity. --->This means the model is underfitting.
#It does not have enough capacity to learn the patterns.


# 4) Reduce Variance:
#    If the model overfits (large train/test gap),
#    add regularization and augmentation.
#This means the model is underfitting.
#It does not have enough capacity to learn the patterns.


# 5) Optimize Training:
#    Tune learning rate, optimizer, and batch size.
# If loss is not decreasing:
# Learning rate might be too low (vanishing gradient ---> not learning much)

# If training unstable: we might be overshooting (lipschitz continuity)


def main():

    # ---------------------------------
    # Step 0: Reproducibility
    # ---------------------------------
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    train_loader, test_loader = get_dataloaders()

    # ---------------------------------
    # Initialize Model
    # ---------------------------------
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    # ---------------------------------
    # GOLDEN RULE STEP 1
    # ---------------------------------
    train_single_sample(model, train_loader, criterion, optimizer, device)

    print("Starting Full Training...\n")

    # ---------------------------------
    # Full Training Loop
    # ---------------------------------
    for epoch in range(Config.EPOCHS):

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch [{epoch+1}/{Config.EPOCHS}]")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test  Accuracy: {test_acc:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    main()