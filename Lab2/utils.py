# utils.py

import torch
import numpy as np
import random


def set_seed(seed: int):
    """
    This function ensures FULL reproducibility.

    Why is this necessary?

    Deep Learning has randomness in:
    - Weight initialization
    - DataLoader shuffling
    - Dropout layers
    - CUDA operations

    Without fixing seeds, every run gives slightly different results.

    In real production experiments, reproducibility is mandatory.
    """

    # Python random
    random.seed(seed)

    # Numpy random
    np.random.seed(seed)

    # PyTorch random (CPU)
    torch.manual_seed(seed)

    # PyTorch random (GPU)
    torch.cuda.manual_seed_all(seed)

    # These two lines make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Global seed set to {seed}")