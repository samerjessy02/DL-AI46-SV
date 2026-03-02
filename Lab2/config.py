# config.py

"""
This file centralizes all hyperparameters and experiment settings.

Why this is important:
- Keeps code clean
- Makes tuning easier
- Improves reproducibility
- Separates experiment configuration from logic
"""


class Config:
    # ===============================
    # Reproducibility
    # ===============================
    # Setting a seed ensures that:
    # - Random initialization is the same
    # - Data shuffling order is the same
    # - Results are reproducible
    SEED = 42

    # ===============================
    # Training Hyperparameters
    # ===============================
    BATCH_SIZE = 128          # Number of samples per gradient update
    EPOCHS = 30               # Number of full passes over the dataset
    LR = 1e-3                 # Learning rate (step size for optimizer)
    WEIGHT_DECAY = 1e-4       # L2 regularization term (helps reduce overfitting)

    # ===============================
    # Data
    # ===============================
    NUM_CLASSES = 10          # CIFAR10 has 10 classes
    DATA_PATH = "./data"      # Where dataset will be downloaded

    # ===============================
    # Device
    # ===============================
    DEVICE = "cuda"           # Will automatically fallback to CPU if unavailable