"""Training script. To run it do: $ python train.py --batch_size=32."""

import argparse
import logging
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import wandb
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import (  # https://lightning.ai/docs/torchmetrics
    MeanMetric,
    SpearmanCorrCoef,
)
from tqdm import tqdm

from druxai._logging import _setup_logger, logger
from druxai.models.NN_minimal import Interaction_Model
from druxai.utils.data import DrugResponseDataset
from druxai.utils.dataframe_utils import (
    split_data_by_cell_line_ids,
    standardize_molecular_data_inplace,
)

# added to not get AttributeError: module '__main__' has no attribute '__spec__'
__spec__ = None


def run(cfg: dict, logger: logging.Logger) -> None:
    """Kick off training."""
    if cfg["WANDB_LOG"]:
        wandb.init(project=cfg["WANDB_PROJECT_NAME"], name=cfg["WANDB_RUN_NAME"], dir=cfg["RESULTS_PATH"], config=cfg)

    # Loss function
    loss_func = nn.HuberLoss()
    torch.manual_seed(cfg["SEED"])
    torch.cuda.manual_seed(cfg["SEED"])

    if not os.path.exists(cfg["RESULTS_PATH"]):
        os.makedirs(cfg["RESULTS_PATH"])

    # Load data
    data = DrugResponseDataset(cfg["DATA_PATH"])
    train_id, val_id, test_id = split_data_by_cell_line_ids(data.targets, seed=cfg["SEED"])
    standardize_molecular_data_inplace(data, train_id, val_id, test_id)

    train_loader = DataLoader(
        data,
        sampler=train_id,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        pin_memory=True,
        num_workers=cfg["NUM_WORKERS"],
        persistent_workers=True,
    )
    val_loader = DataLoader(
        data,
        sampler=val_id,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        pin_memory=True,
        num_workers=cfg["NUM_WORKERS"],
        persistent_workers=True,
    )

    # Determine whether to initialize a new model from scratch or resume training from a checkpoint
    best_val_loss = 1e9
    iter_num = 0
    if cfg["INIT_FROM"] == "scratch":
        logger.info("Training model from scratch.")
        model = Interaction_Model(data)
        model.train().to(cfg["DEVICE"])
        # Setup optimizers
        optimizer1 = AdamW(
            model.nn1.parameters(),
            cfg["LEARNING_RATE"],
            betas=(cfg["BETA1"], cfg["BETA2"]),
            weight_decay=cfg["WEIGHT_DECAY"],
        )
        optimizer2 = AdamW(
            model.nn2.parameters(),
            cfg["LEARNING_RATE"],
            betas=(cfg["BETA1"], cfg["BETA2"]),
            weight_decay=cfg["WEIGHT_DECAY"],
        )
    elif cfg["INIT_FROM"] == "resume":
        logger.info(f"Resuming training from {cfg['RESULTS_PATH']}/ckpt.pt")
        # Load checkpoint
        ckpt_path = os.path.join(cfg["RESULTS_PATH"], "ckpt.pt")
        checkpoint = torch.load(ckpt_path)
        # Initialize model with the same configuration as the checkpoint
        model = Interaction_Model(data)
        model.train().to(cfg["DEVICE"])
        model.load_state_dict(checkpoint["model"])
        # Setup optimizers
        optimizer1 = AdamW(
            model.nn1.parameters(),
            cfg["LEARNING_RATE"],
            betas=(cfg["BETA1"], cfg["BETA2"]),
            weight_decay=cfg["WEIGHT_DECAY"],
        )
        optimizer2 = AdamW(
            model.nn2.parameters(),
            cfg["LEARNING_RATE"],
            betas=(cfg["BETA1"], cfg["BETA2"]),
            weight_decay=cfg["WEIGHT_DECAY"],
        )
        # Also load optimizer states if needed
        optimizer1.load_state_dict(checkpoint["optimizer 1"])
        optimizer2.load_state_dict(checkpoint["optimizer 2"])
        # Update best validation loss
        best_val_loss = checkpoint["best_val_loss"]
        iter_num = checkpoint["iter_num"]
        model.train().to(cfg["DEVICE"])
    else:
        raise ValueError("init_from must be either 'scratch' or 'resume'")

    # Set Learning Rate scheduler
    scheduler1 = ExponentialLR(optimizer1, gamma=0.9)
    scheduler2 = ExponentialLR(optimizer2, gamma=0.9)

    train(
        model=model,
        loss_func=loss_func,
        optimizers=(optimizer1, optimizer2),
        schedulers=(scheduler1, scheduler2),
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
    )

    # Finish the wandb run
    if cfg["WANDB_LOG"]:
        wandb.finish()


def train(
    model: nn.Module,
    loss_func: nn.Module,
    optimizers: Tuple[Optimizer, Optimizer],
    schedulers: Tuple[_LRScheduler, _LRScheduler],
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    iter_num: int = 0,
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
        iter_num (int, optional): The current iteration number. Defaults to 0.
        best_val_loss (int, optional): The best validation loss obtained during training. Defaults to 1e9.
    """
    metric_train_loss = MeanMetric().to(cfg["DEVICE"])
    metric_val_loss = MeanMetric().to(cfg["DEVICE"])
    metric_train_rscore = SpearmanCorrCoef().to(cfg["DEVICE"])
    metric_val_rscore = SpearmanCorrCoef().to(cfg["DEVICE"])

    optimizer1, optimizer2 = optimizers[0], optimizers[1]
    scheduler1, scheduler2 = schedulers[0], schedulers[1]
    for epoch in range(cfg["EPOCHS"]):
        model.train()
        for X, y, _ in tqdm(train_loader, leave=False):
            drug, molecular, outcome = (
                X["drug_encoding"].to(cfg["DEVICE"]),
                X["gene_expression"].to(cfg["DEVICE"]),
                y.to(cfg["DEVICE"]),
            )

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            prediction = model.forward(drug, molecular)
            loss = loss_func(prediction, outcome)
            metric_train_loss(loss)
            metric_train_rscore(prediction, outcome)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP"])
            # Randomly select optimizer
            selected_optimizer = optimizer1 if torch.rand(1) < 0.5 else optimizer2
            selected_optimizer.step()

            if (iter_num % cfg["EVAL_INTERVAL"] == 0) & (iter_num != 0):
                # ------------------------------------------------------------------------------
                # TEST LOOP
                # ------------------------------------------------------------------------------
                model.eval()
                for X, y, _ in tqdm(val_loader, leave=False):
                    drug, molecular, outcome = (
                        X["drug_encoding"].to(cfg["DEVICE"]),
                        X["gene_expression"].to(cfg["DEVICE"]),
                        y.to(cfg["DEVICE"]),
                    )

                    with torch.no_grad():
                        prediction = model(drug, molecular)
                        batch_loss = loss_func(outcome, prediction)

                        metric_val_loss(batch_loss)
                        metric_val_rscore(prediction, outcome)

                # Compute epoch metrics from cached batch metrics. Take absolute values of R Scores for eval!
                train_loss = float(metric_train_loss.compute())
                val_loss = float(metric_val_loss.compute())
                train_rscore = float(abs(metric_train_rscore.compute()))
                val_rscore = float(abs(metric_val_rscore.compute()))

                # Reset metrics for next evaluation interval
                metric_train_loss.reset()
                metric_val_loss.reset()
                metric_train_rscore.reset()
                metric_val_rscore.reset()
                if cfg["WANDB_LOG"]:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "epoch": epoch,
                            "train loss": train_loss,
                            "val_loss": val_loss,
                            "r2_train": train_rscore,
                            "r2_val": val_rscore,
                            "lr_opt1": optimizer1.param_groups[0]["lr"],
                            "lr_opt2": optimizer2.param_groups[0]["lr"],
                        }
                    )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer 1": optimizer1.state_dict(),
                        "optimizer 2": optimizer2.state_dict(),
                        "iter_num": iter_num,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "config": cfg,
                    }
                    logger.info(
                        f"New best val_loss achieved! \n"
                        f"Epoch: {epoch} "
                        f"Train Loss: {train_loss:.4f}; Train R2: {train_rscore:.4f} "
                        f"Val Loss: {best_val_loss:.4f}; Val R2: {val_rscore:.4f} \n"
                        f"Saving checkpoint to {os.path.join(cfg['RESULTS_PATH'], 'ckpt.pt')}"
                    )
                    torch.save(checkpoint, os.path.join(cfg["RESULTS_PATH"], "ckpt.pt"))
            iter_num += 1

        if cfg["DECAY_LR"]:
            scheduler1.step()
            scheduler2.step()


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
    # Get correct config file over command line.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/results/training/config.yml",
        type=str,
    )
    args = parser.parse_args()

    # Read yml file to memory as dict.
    cfg = read_yml(args.config)

    logger = _setup_logger(log_path=cfg["RESULTS_PATH"])

    # Get information about allocated GPUs
    check_cuda_devices(logger)

    # Start training with chosen configuration.
    run(cfg, logger)


if __name__ == "__main__":
    main()
