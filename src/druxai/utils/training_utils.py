"""Functions which modularize the train.py file."""

import os
from typing import Dict, Tuple

import torch
import yaml
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import (  # https://lightning.ai/docs/torchmetrics
    MeanMetric,
    SpearmanCorrCoef,
)

from druxai.models.NN_minimal import Interaction_Model
from druxai.utils.data import DataloaderSampler, Dataset


def build_dataloader(
    data: Dataset, train_sampler: DataloaderSampler, val_sampler: DataloaderSampler, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Build data loaders for training and validation.

    Parameters
    ----------
        data (Dataset): Dataset containing the training and validation data.
        train_sampler (DataloaderSampler): Sampler for training data.
        val_sampler (DataloaderSampler): Sampler for validation data.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): How many subprocesses to use for data loading.

    Returns
    -------
        tuple: Tuple containing the training and validation data loaders.
    """
    # Creating DataLoader for training data
    train_loader = DataLoader(
        data,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,  # Setting pin_memory to True for faster data transfer
        num_workers=num_workers,
        persistent_workers=True,
    )

    # Creating DataLoader for validation data
    val_loader = DataLoader(
        data,
        sampler=val_sampler,
        batch_size=batch_size,
        pin_memory=True,  # Setting pin_memory to True for faster data transfer
        num_workers=num_workers,
        persistent_workers=True,
    )

    return train_loader, val_loader


def set_optimizers(
    model: Interaction_Model, optimizer: str, learning_rate: float, weight_decay: float = 0
) -> Tuple[Optimizer, Optimizer]:
    """
    Build optimizers for the model.

    Parameters
    ----------
        model (Interaction_Model): The interaction model to optimize.
        optimizer (str): The type of optimizer to use. It should be either 'sgd' or 'adam'.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The L2 penalty. (default: 0)

    Returns
    -------
        tuple: Tuple containing the optimizers for the model.
    """
    # Initializing optimizers based on the selected optimizer
    if optimizer == "sgd":
        optimizer1 = SGD(model.nn1.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        optimizer2 = SGD(model.nn2.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer1 = Adam(model.nn1.parameters(), lr=learning_rate)
        optimizer2 = Adam(model.nn2.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Please choose 'sgd' or 'adam'.")

    return optimizer1, optimizer2


def set_schedulers(scheduler: str, optimizer1: Optimizer, optimizer2: Optimizer) -> Tuple[_LRScheduler, _LRScheduler]:
    """
    Set learning rate schedulers for the optimizers.

    Parameters
    ----------
    scheduler : str
        The type of scheduler to use.
    optimizer1 : Optimizer
        The first optimizer.
    optimizer2 : Optimizer
        The second optimizer.

    Returns
    -------
    tuple
        Tuple containing two learning rate schedulers for the optimizers.
    """
    if scheduler == "exponential":
        scheduler1 = ExponentialLR(optimizer1, gamma=0.9)
        scheduler2 = ExponentialLR(optimizer2, gamma=0.9)
    elif scheduler == "plateau":
        scheduler1 = ReduceLROnPlateau(optimizer1, mode="min", factor=0.1, patience=10)
        scheduler2 = ReduceLROnPlateau(optimizer2, mode="min", factor=0.1, patience=10)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}.")

    return scheduler1, scheduler2


def set_loss(loss_type: str, *args, **kwargs) -> Tuple:
    """
    Set loss function(s) based on the specified type.

    Parameters
    ----------
    loss_type : str
        The type of loss function to use.

    Returns
    -------
    tuple
        Tuple containing the required loss function(s).
    """
    if loss_type == "mse":
        # Assuming you're using mean squared error loss
        loss_func = torch.nn.MSELoss(*args, **kwargs)
    elif loss_type == "huber":
        # Huber loss
        loss_func = torch.nn.SmoothL1Loss(*args, **kwargs)
    else:
        raise ValueError(f"Unknown loss function type: {loss_type}.")

    return loss_func


def build_network():
    """Create Network with different configuartion options."""
    pass


def evaluate(
    model: torch.nn.Module, val_loader: DataLoader, loss_func: torch.nn.Module, device: torch.device
) -> tuple[float, float]:
    """
    Evaluate the given model on the validation data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation data.
        loss_func (torch.nn.Module): Loss function to evaluate the performance.
        device (torch.device): Device on which to perform the evaluation.

    Returns
    -------
        tuple[float, float]: Validation loss and Spearman correlation coefficient score.
    """
    model.eval()
    metric_val_loss = MeanMetric().to(device)
    metric_val_rscore = SpearmanCorrCoef().to(device)

    with torch.no_grad():
        for X, y, _ in val_loader:
            drug, molecular, outcome = (
                X["drug_encoding"].to(device),
                X["gene_expression"].to(device),
                y.to(device),
            )

            prediction = model(drug, molecular)
            val_loss = loss_func(prediction, outcome)
            metric_val_loss(val_loss)
            metric_val_rscore(prediction, outcome)

    val_loss = float(metric_val_loss.compute())
    val_rscore = float(metric_val_rscore.compute())

    # Reset metrics for the next evaluation
    metric_val_loss.reset()
    metric_val_rscore.reset()

    return val_loss, val_rscore


def save_checkpoint(model, optimizer1, optimizer2, iter_num, best_val_loss, fixed_cfg, logger):
    """
    Save the model checkpoint.

    Parameters
    ----------
        model: The model to save.
        optimizer1: Optimizer for the first part of the model.
        optimizer2: Optimizer for the second part of the model.
        iter_num: Current iteration number.
        best_val_loss: Best validation loss achieved during training.
        fixed_cfg: Fixed configuration parameters.
        logger: Logger for logging.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer1": optimizer1.state_dict(),
        "optimizer2": optimizer2.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": fixed_cfg,
    }
    checkpoint_path = os.path.join(fixed_cfg["RESULTS_PATH"], "ckpt.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(
        f"New best val_loss achieved! \n" f"Val Loss: {best_val_loss:.4f}\n" f"Saving checkpoint to {checkpoint_path}"
    )


def load_yaml(filename: str) -> Dict:
    """
    Load data from a YAML file into a Python dictionary.

    Args:
        filename (str): The path to the YAML file.

    Returns
    -------
        dict: A dictionary containing the data loaded from the YAML file.
    """
    with open(filename) as file:
        return yaml.safe_load(file)


def get_ops_device_string():
    """
    Return the string representation of the appropriate torch device based on availability of CUDA and MPS.

    Priority: cuda > mps > cpu
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
