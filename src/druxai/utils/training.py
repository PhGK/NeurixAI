"""Method which modularize train.py file."""

from typing import Tuple

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from druxai.utils.data import DataloaderSampler, DrugResponseDataset


def build_dataloaders(
    data: DrugResponseDataset,
    train_sampler: DataloaderSampler,
    val_sampler: DataloaderSampler,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build data loaders for training and validation datasets.

    Args:
        data (DrugResponseDataset): The dataset containing drug response data.
        train_sampler (DataloaderSampler): The sampler to use for sampling data during training.
        val_sampler (DataloaderSampler): The sampler to use for sampling data during validation.
        batch_size (int): The number of samples per batch to load.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns
    -------
        Tuple[DataLoader, DataLoader]: A tuple containing the training and validation data loaders.
    """
    train_loader = DataLoader(
        data,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        data,
        sampler=val_sampler,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    return train_loader, val_loader


def build_optimizer(network, optimizer, learning_rate):
    """Build optmizers of choice."""
    if optimizer == "sgd":
        optimizer = SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = Adam(network.parameters(), lr=learning_rate)
    return optimizer
