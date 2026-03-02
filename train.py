# train.py

import torch


def train_single_sample(model, loader, criterion, optimizer, device):
    """
    GOLDEN RULE STEP 1 — SANITY CHECK

    We train on ONE single image only.

    If the model cannot achieve 100% accuracy on one image,
    then something is wrong in:
    - Model architecture
    - Loss function
    - Backprop
    - Optimizer

    This is the most important debugging step in Deep Learning.
    """

    model.train()

    data, target = next(iter(loader))

    # Take only ONE sample
    data = data[0:1].to(device)
    target = target[0:1].to(device)

    print("\n[Sanity Check] Training on ONE sample...")

    for epoch in range(200):

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss {loss.item():.4f} | Acc {acc.item():.4f}")

    print("If accuracy reaches 1.0 → Pipeline is correct ✅\n")


def train_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_correct = 0
    total_loss = 0

    for data, target in loader:

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_correct += (output.argmax(1) == target).sum().item()

    accuracy = total_correct / len(loader.dataset)

    return total_loss / len(loader), accuracy


def evaluate(model, loader, device):

    model.eval()
    total_correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_correct += (output.argmax(1) == target).sum().item()

    return total_correct / len(loader.dataset)