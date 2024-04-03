"""Training script. To run it do: $ python train.py."""

from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import (  # https://lightning.ai/docs/torchmetrics
    MeanMetric,
    SpearmanCorrCoef,
)
from tqdm import tqdm

import wandb
from druxai._logging import _setup_logger
from druxai.utils.data import DataloaderSampler, DrugResponseDataset
from druxai.utils.dataframe_utils import (
    k_fold_split_data_by_cell_lines_ids,
    standardize_molecular_data_inplace,
)
from druxai.utils.set_seeds import set_seeds
from druxai.utils.training_utils import (
    build_dataloader,
    check_cuda_devices,
    evaluate,
    get_ops_device_string,
    load_yaml,
    save_checkpoint,
    setup_training,
)

# added to not get AttributeError: module '__main__' has no attribute '__spec__'
__spec__ = None

# SET THIS PATH TO YOUR FIXED CONFIG FILE DIRECTORY
fixed_cfg = load_yaml("/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/fixed_config.yml")

# SET THESE CONFIG PARAMETERS
config = {
    "name": "cross_val",
    "project": "Cross Validation Training DruxAI",
    "metric": {"name": "r2_val", "goal": "maximize"},
    "nsplits": 5,
    "resume": False,
    "patience": 5,
    "epochs": 1,
    "optimizer": "sgd",
    "scheduler": "exponential",
    "loss": "huber",
    "batch_size": 128,
    "learning_rate": 0.1,
    "output_features": 256,
    "hidden_dims_nn1": [512],
    "hidden_dims_nn2": [100],
    "dropout_nn1": 0.1,
    "dropout_nn2": 0.1,
}

# Create logger
logger = _setup_logger(log_path=fixed_cfg["RESULTS_PATH"])

# Set device
device = get_ops_device_string()

# Set seeds
set_seeds()

# Create Namespace for config
config = SimpleNamespace(**config)


def run(config=None) -> None:
    """Kick off training."""
    # Load data
    data = DrugResponseDataset(fixed_cfg["DATA_PATH"])

    train_ids, test_ids = k_fold_split_data_by_cell_lines_ids(data.targets, n_splits=config.nsplits)

    for i in range(config.nsplits):
        with wandb.init(
            project=config.project,
            config=config,
            name=f"{config.name}_fold_{i+1}",
            dir=fixed_cfg["RESULTS_PATH"],
        ):
            train_id, test_id = train_ids[i], test_ids[i]
            train_sampler, val_sampler = DataloaderSampler(train_id), DataloaderSampler(test_id)
            standardize_molecular_data_inplace(data, train_id=train_id, test_id=test_id, cross_val=True)
            logger.info("Finished Loading Data")

            train_loader, val_loader = build_dataloader(
                data, train_sampler, val_sampler, config.batch_size, fixed_cfg["NUM_WORKERS"]
            )

            model, optimizer1, optimizer2, scheduler1, scheduler2, loss_func, best_val_loss = setup_training(
                config, data, logger, fixed_cfg, device
            )

            train(
                model=model,
                loss_func=loss_func,
                optimizers=(optimizer1, optimizer2),
                schedulers=(scheduler1, scheduler2),
                train_loader=train_loader,
                val_loader=val_loader,
                fixed_cfg=fixed_cfg,
                device=device,
                best_val_loss=best_val_loss,
                epochs=config.epochs,
            )


def train(
    model: nn.Module,
    loss_func: nn.Module,
    optimizers: Tuple[Optimizer, Optimizer],
    schedulers: Tuple[_LRScheduler, _LRScheduler],
    train_loader: DataLoader,
    val_loader: DataLoader,
    fixed_cfg: Dict,
    device,
    epochs,
    best_val_loss: int = 1e9,
):
    """Train DruxAI Network end-to-end.

    Args:
        model (nn.Module): The neural network model to be trained.
        loss_func (nn.Module): The loss function used for training.
        optimizers (Tuple): A tuple containing two optimizers for the model's parameters.
        schedulers (Tuple): A tuple containing two learning rate schedulers corresponding to the optimizers.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        cfg (Dict): A dictionary containing configuration parameters for training.
        device (str or torch.device, optional): The device to run the computations on ('cpu' or 'cuda').
        epochs (int): Number of epochs to train the model for.
        best_val_loss (int, optional): The best validation loss obtained during training. Defaults to 1e9.
    """
    optimizer1, optimizer2 = optimizers[0], optimizers[1]
    scheduler1, scheduler2 = schedulers[0], schedulers[1]
    metric_train_loss = MeanMetric().to(device)
    metric_train_rscore = SpearmanCorrCoef().to(device)

    # Track gradients
    wandb.watch(model)
    early_stopping_patience = config.patience  # Number of epochs to wait before stopping if no improvement
    epochs_without_improvement = 0  # Keep track of the number of epochs without improvement

    for epoch in range(epochs):

        model.train()
        for X, y, _ in tqdm(train_loader, leave=False):
            drug, molecular, outcome = (
                X["drug_encoding"].to(device),
                X["gene_expression"].to(device),
                y.to(device),
            )

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            prediction = model(drug, molecular)
            train_loss = loss_func(prediction, outcome)

            metric_train_loss(train_loss)
            metric_train_rscore(prediction, outcome)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fixed_cfg["GRAD_CLIP"])
            selected_optimizer = optimizer1 if torch.rand(1) < 0.5 else optimizer2
            selected_optimizer.step()

        avg_train_loss = metric_train_loss.compute()
        train_rscore = metric_train_rscore.compute()
        val_loss, val_rscore = evaluate(model, val_loader, loss_func, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset the counter
            save_checkpoint(model, optimizer1, optimizer2, epoch, best_val_loss, fixed_cfg, logger, save_model=False)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    f"No improvement in validation loss for {early_stopping_patience} epochs. Stopping training."
                )
                break

        wandb.log(
            {
                "epoch": epoch,
                "train loss": avg_train_loss,
                "val_loss": val_loss,
                "r2_train": train_rscore,
                "r2_val": val_rscore,
                "lr_opt1": optimizer1.param_groups[0]["lr"],
                "lr_opt2": optimizer2.param_groups[0]["lr"],
            }
        )

        metric_train_loss.reset()
        metric_train_rscore.reset()
        scheduler1.step()
        scheduler2.step()

    wandb.finish()


def main() -> None:
    """Execute main func."""
    # Get information about allocated GPUs
    check_cuda_devices(logger)

    # Run training
    run(config=config)


if __name__ == "__main__":
    main()
