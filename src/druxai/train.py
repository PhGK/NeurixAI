"""Training script. To run it do: $ python train.py --batch_size=32."""

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import yaml
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
from druxai.models.NN_flexible import Interaction_Model
from druxai.utils.data import DataloaderSampler, DrugResponseDataset
from druxai.utils.dataframe_utils import (
    split_data_by_cell_line_ids,
    standardize_molecular_data_inplace,
)
from druxai.utils.set_seeds import set_seeds
from druxai.utils.training_utils import (
    build_dataloader,
    evaluate,
    get_ops_device_string,
    load_yaml,
    set_loss,
    set_optimizers,
    set_schedulers,
)

# added to not get AttributeError: module '__main__' has no attribute '__spec__'
__spec__ = None

# SET THIS PATH TO YOUR FIXED CONFIG FILE DIRECTORY
fixed_cfg = load_yaml("/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/fixed_config.yml")

# Create logger
logger = _setup_logger(log_path=fixed_cfg["RESULTS_PATH"])

# Set device
device = get_ops_device_string()

# Set seeds
set_seeds()

sweep_config = {
    "method": "bayes",
    "metric": {"name": "r2_val", "goal": "maximize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
        "max_iter": 50,
    },
    "parameters": {
        "optimizer": {"values": ["sgd", "adam"]},
        "scheduler": {"values": ["exponential"]},
        "loss": {"values": ["huber", "mse"]},
        "epochs": {"values": [5, 10, 25, 50]},
        "batch_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [0.1, 0.01, 0.05, 0.001]},
        "output_features": {"value": 1000},
        "hidden_dims_nn1": {"values": [[5000], [10000], [1000, 100], [128, 64], [256, 128]]},
        "hidden_dims_nn2": {"values": [[10000], [5000], [1000, 100], [128, 64], [256, 128]]},
        "dropout_nn1": {"values": [0.1, 0.2, 0.3]},
        "dropout_nn2": {"values": [0.1, 0.2, 0.3]},
    },
}


def run(config=None) -> None:
    """Kick off training."""
    with wandb.init(
        config=config,
        dir=fixed_cfg["RESULTS_PATH"],
    ):
        config = wandb.config  # this config will be set by Sweep Controller

        if not os.path.exists(fixed_cfg["RESULTS_PATH"]):
            os.makedirs(fixed_cfg["RESULTS_PATH"])

        # Load data
        data = DrugResponseDataset(fixed_cfg["DATA_PATH"])
        train_id, val_id, test_id = split_data_by_cell_line_ids(data.targets, seed=1337)
        train_sampler, val_sampler = DataloaderSampler(train_id), DataloaderSampler(val_id)
        standardize_molecular_data_inplace(data, train_id, val_id, test_id)
        logger.info("Finished Loading Data")

        train_loader, val_loader = build_dataloader(
            data, train_sampler, val_sampler, config.batch_size, fixed_cfg["NUM_WORKERS"]
        )

        best_val_loss = 1e9
        model = Interaction_Model(
            data,
            config.output_features,
            config.hidden_dims_nn1,
            config.hidden_dims_nn2,
            config.dropout_nn1,
            config.dropout_nn2,
        )
        model.train().to(device)
        optimizer1, optimizer2 = set_optimizers(model, config.optimizer, config.learning_rate)

        # Set Learning Rate scheduler and loss function
        scheduler1, scheduler2 = set_schedulers(config.scheduler, optimizer1, optimizer2)
        loss_func = set_loss(config.loss)

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
            logger.info(f"New best val_loss achieved! \n" f"Epoch: {epoch}; Val Loss: {best_val_loss:.4f}")
            # save_checkpoint(model, optimizer1, optimizer2, iter_num, best_val_loss, fixed_cfg, logger)

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


def main() -> None:
    """Execute main func."""
    # Get information about allocated GPUs
    check_cuda_devices(logger)

    # Start a sweep with the defined sweep configuration
    sweep_id = wandb.sweep(sweep_config, project=fixed_cfg["SWEEPNAME"])

    # Run the sweep
    wandb.agent(sweep_id, function=run, count=2)


if __name__ == "__main__":
    main()
