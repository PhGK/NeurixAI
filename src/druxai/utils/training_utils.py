"""Functions which modularize the train.py file."""

import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch.nn import Module
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import (  # https://lightning.ai/docs/torchmetrics
    MeanMetric,
    SpearmanCorrCoef,
)

import pandas as pd

from druxai.models.NN_flexible import Interaction_Model
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
        optimizer1 = SGD(model.drug_nn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        optimizer2 = SGD(
            model.gene_expression_nn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    elif optimizer == "adam":
        optimizer1 = Adam(model.drug_nn.parameters(), lr=learning_rate)
        optimizer2 = Adam(model.gene_expression_nn.parameters(), lr=learning_rate)
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
        loss_func = torch.nn.HuberLoss(*args, **kwargs)
    else:
        raise ValueError(f"Unknown loss function type: {loss_type}.")

    return loss_func


def build_network():
    """Create Network with different configuartion options."""
    pass


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_func: torch.nn.Module,
    device: torch.device,
    fixed_cfg: Optional[Dict[str, Any]] = None,
    prediction_file: str = None,
) -> tuple[float, float]:
    """
    Evaluate the given model on the validation data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation data.
        loss_func (torch.nn.Module): Loss function to evaluate the performance.
        device (torch.device): Device on which to perform the evaluation.
        fixed_cfg (Optional[Dict[str, Any]], optional): Configuration parameters used for accessing result path.
                                                        Defaults to None.
        prediction_file (str, optional): The file to save the predictions to. Defaults to None.

    Returns
    -------
        tuple[float, float]: Validation loss and Spearman correlation coefficient score.
    """
    model.eval()
    metric_val_loss = MeanMetric().to(device)
    metric_val_rscore = SpearmanCorrCoef().to(device)

    with torch.no_grad():
        out, pred, ids = [], [], []
        for X, y, idx in val_loader:
            drug, molecular, outcome = (
                X["drug_encoding"].to(device),
                X["gene_expression"].to(device),
                y.to(device),
            )

            prediction = model(drug, molecular)
            val_loss = loss_func(prediction, outcome)
            metric_val_loss(val_loss)
            metric_val_rscore(prediction, outcome)

            # Write the results to the lists
            out.extend(outcome.squeeze(1).cpu().tolist())
            pred.extend(prediction.squeeze(1).cpu().tolist())
            ids.extend(idx.tolist())

        df = pd.DataFrame(
            {
                "ground_truth": out,
                "prediction": pred,
                "cells": val_loader.dataset.targets.iloc[ids]["cell_line"],
                "drugs": val_loader.dataset.targets.iloc[ids]["DRUG"],
            }
        )

        if prediction_file is not None:
            assert fixed_cfg is not None, "Fixed configuration parameters are required to save the predictions."
            save_dir = os.path.join(fixed_cfg["RESULTS_PATH"], "predictions")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, prediction_file)
            df.to_csv(save_path, index=False)

    val_loss = float(metric_val_loss.compute())
    val_rscore = float(metric_val_rscore.compute())

    # Reset metrics for the next evaluation
    metric_val_loss.reset()
    metric_val_rscore.reset()

    return val_loss, val_rscore


def save_checkpoint(
    model: Module,
    optimizer1: Optimizer,
    optimizer2: Optimizer,
    epoch: int,
    best_val_loss: float,
    fixed_cfg: Dict[str, Any],
    logger: logging.Logger,
    save_model: bool = True,
):
    """
    Save the model checkpoint.

    Parameters
    ----------
        model: The model to save.
        optimizer1: Optimizer for the first part of the model.
        optimizer2: Optimizer for the second part of the model.
        epoch: Current epoch number.
        best_val_loss: Best validation loss achieved during training.
        fixed_cfg: Fixed configuration parameters.
        logger: Logger for logging.
        save_model: Whether to save the model or not.
    """
    if save_model:
        checkpoint = {
            "model": model.state_dict() if save_model else None,
            "optimizer1": optimizer1.state_dict(),
            "optimizer2": optimizer2.state_dict(),
            "iter_num": epoch,
            "best_val_loss": best_val_loss,
            "config": fixed_cfg,
        }
        checkpoint_path = os.path.join(fixed_cfg["RESULTS_PATH"], "ckpt.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(
            f"Epoch: {epoch} new best val_loss: {best_val_loss:.4f}\n" f"Saving checkpoint to {checkpoint_path}"
        )

    logger.info(f"Epoch: {epoch} new best val_loss: {best_val_loss:.4f}\n")


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


def setup_training(
    cfg: Namespace, data: Any, logger: Any, fixed_cfg: dict, device: torch.device
) -> Tuple[Interaction_Model, SGD, SGD, float, int]:
    """
    Set up the training process based on the configuration and preloads weights if needed.

    Args:
        cfg (Namespace): The configuration namespace.
        data (Any): The data to be used for training.
        logger (Any): The logger to be used for logging information.
        fixed_cfg (dict): The fixed configuration dictionary.
        device (torch.device): Device on which to perform the evaluation.

    Returns
    -------
        Tuple[Interaction_Model, SGD, SGD, float, int]: Returns the model, two optimizers, the best validation loss,
                                                        and the iteration number.
    """
    if not cfg.resume:
        logger.info("Training model from scratch.")
        model = Interaction_Model(
            data,
            cfg.output_features,
            cfg.hidden_dims_drug_nn,
            cfg.hidden_dims_gene_expression_nn,
            cfg.dropout_drug_nn,
            cfg.dropout_gene_expression_nn,
        )
        model.train().to(device)
        optimizer1, optimizer2 = set_optimizers(model, cfg.optimizer, cfg.learning_rate)
        scheduler1, scheduler2 = set_schedulers(cfg.scheduler, optimizer1, optimizer2)
        loss_func = set_loss(cfg.loss)
        best_val_loss = 1e9
    else:
        logger.info(f"Resuming training from {fixed_cfg['RESULTS_PATH']}/ckpt.pt")
        # Load checkpoint
        ckpt_path = os.path.join(fixed_cfg["RESULTS_PATH"], "ckpt.pt")
        checkpoint = torch.load(ckpt_path)
        # Initialize model with the same configuration as the checkpoint
        model = Interaction_Model(
            data,
            cfg.output_features,
            cfg.hidden_dims_drug_nn,
            cfg.hidden_dims_gene_expression_nn,
            cfg.dropout_drug_nn,
            cfg.dropout_gene_expression_nn,
        )
        model.load_state_dict(checkpoint["model"])
        model.train().to(device)
        optimizer1, optimizer2 = set_optimizers(model, cfg.optimizer, cfg.learning_rate)
        scheduler1, scheduler2 = set_schedulers(cfg.scheduler, optimizer1, optimizer2)
        loss_func = set_loss(cfg.loss)
        # Also load optimizer states if needed
        optimizer1.load_state_dict(checkpoint["optimizer1"])
        optimizer2.load_state_dict(checkpoint["optimizer2"])
        # Update best validation loss
        best_val_loss = checkpoint["best_val_loss"]

    return model, optimizer1, optimizer2, scheduler1, scheduler2, loss_func, best_val_loss


def check_cuda_devices(logger):
    """Print information about available GPUs."""
    num_devices = torch.cuda.device_count()
    logger.info("-" * 50)
    if num_devices > 1:
        logger.info(f"{num_devices} GPUs are available:")
    elif num_devices == 1:
        logger.info("1 GPU is available:")
    else:
        logger.info("WARNING! NO GPU DETECTED!")
        logger.info("-" * 50)
        return

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"{i} | {props.name:<20} | {props.total_memory / 1024**3:2.2f} GB Memory")
    logger.info("-" * 50)


def read_yml(filepath: str) -> dict:
    """Load a yml file to memory as dict."""
    with open(filepath) as ymlfile:
        return dict(yaml.load(ymlfile, Loader=yaml.FullLoader))


def save_results_to_disk(results: List[float], filename: str) -> None:
    """
    Save the results to a CSV file.

    Parameters
    ----------
    results (List[float]): A list of predictions.
    filename (str): The name of the file to save the results to.

    Returns
    -------
    None
    """
    # Convert the results to a DataFrame
    df = pd.DataFrame(results, columns=["Prediction"])

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
