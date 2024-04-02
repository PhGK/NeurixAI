"""Set the seed automatically for pytorch, random and numpy packages for deterministic behavior."""

import random

import torch

import numpy as np


def set_seeds(seed: int = 42):
    """
    Set seeds for torch, numpy, and random modules for reproducibility.

    Parameters
    ----------
        seed (int): Seed value to set. Default is 42.

    Returns
    -------
        None
    """
    torch.manual_seed(seed)  # Set seed for torch
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Set deterministic mode for cuDNN
    np.random.default_rng(seed)  # Set seed for numpy
    random.seed(seed)  # Set seed for random module
