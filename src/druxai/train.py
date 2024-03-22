"""Training script. To run it do: $ python train.py --batch_size=32."""

import argparse
import logging
import os
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import wandb
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd

from druxai._logging import _setup_logger, logger
from druxai.models.NN_minimal import Interaction_Model
from druxai.utils.data import DrugResponseDataset
from druxai.utils.dataframe_utils import (
    create_batch_result,
    split_data_by_cell_line_ids,
    standardize_molecular_data_inplace,
)

# added to not get AttributeError: module '__main__' has no attribute '__spec__'
__spec__ = None


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader_val: DataLoader,
    data: DrugResponseDataset,
    epoch: int,
    cfg: Dict,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    save_evaluation: bool = True,
    eval_metric: str = "spearmanr",
) -> Tuple[float, float]:
    """Evaluate the performance of a neural network model on a validation dataset.

    This function evaluates the given `model` on the provided `val_data` using the `dataloader_val`.
    It calculates the average loss and the Pearson correlation coefficient (r score) for the predictions.

    Args:
        model (torch.nn.Module): The neural network model being evaluated.
        dataloader_val (DataLoader): DataLoader containing validation data.
        val_data (MyDataSet): Validation dataset.
        epoch (int): The current epoch number.
        save_evaluation (bool, optional): Whether to save the final predictions. Defaults to True.
        eval_metric (str, optional): The evaluation metric to use, either "spearmanr" or "pearsonr".
                                    Defaults to "spearmanr".

    Returns
    -------
        Tuple[float, float]: A tuple containing the average loss and the Pearson correlation coefficient (r score).

    Example:
        avg_loss, r_score = evaluate_model(model, dataloader_val, val_data, 5, save_evaluation=True)
    """
    model.eval()
    prediction_frames = []
    all_outcomes = []
    all_predictions = []
    losses = torch.zeros(len(dataloader_val))
    for iter_count, (X, y, idx) in tqdm(enumerate(dataloader_val)):
        drug, molecular, outcome = (
            X["drug_encoding"].to(cfg["DEVICE"]),
            X["gene_expression"].to(cfg["DEVICE"]),
            y.to(cfg["DEVICE"]),
        )

        prediction = model(drug, molecular)
        loss = loss_func(outcome, prediction)
        losses[iter_count] = loss.item()

        all_outcomes.extend(outcome.cpu().detach().numpy().reshape(-1))
        all_predictions.extend(prediction.cpu().detach().numpy().reshape(-1))
        if epoch == (cfg["EPOCHS"] - 1):
            batch_result = create_batch_result(outcome, prediction, data, idx, epoch)
            prediction_frames.append(batch_result)

    if save_evaluation & (epoch == (cfg["EPOCHS"] - 1)):
        pd.concat(prediction_frames, axis=0).to_csv(os.path.join(cfg["WEIGHT_DECAY"], "prediction.csv"))

    if eval_metric == "spearmanr":
        r_score, _ = spearmanr(all_outcomes, all_predictions)
    elif eval_metric == "pearsonr":
        r_score, _ = pearsonr(all_outcomes, all_predictions)
    else:
        raise ValueError("Invalid evaluation metric. Use 'spearmanr' or 'pearsonr'.")

    avg_loss = losses.mean()
    model.train()
    return avg_loss, r_score


def run(cfg: dict, logger: logging.Logger) -> None:
    """Kick off training."""
    if cfg["WANDB_LOG"]:
        wandb.init(
            project=cfg["WANDB_PROJECT_NAME"],
            name=cfg["WANDB_RUN_NAME"],
            config=cfg,
        )

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
        logger.info(f"Resuming training from {cfg['WEIGHT_DECAY']}/ckpt.pt")
        # Load checkpoint
        ckpt_path = os.path.join(cfg["WEIGHT_DECAY"], "ckpt.pt")
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
    scheduler2 = ExponentialLR(optimizer1, gamma=0.9)
    train(
        model=model,
        loss_func=loss_func,
        optimizers=(optimizer1, optimizer2),
        schedulers=(scheduler1, scheduler2),
        train_loader=train_loader,
        val_loader=val_loader,
        data=data,
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
    data: DrugResponseDataset,
    cfg: Dict,
    iter_num: int = 0,
    best_val_loss: int = 1e9,
):
    """Train DruxAI Network end-to-end.

    Args:
        model (nn.Module): _description_
        loss_func (nn.Module): _description_
        optimizers (Tuple): _description_
        schedulers (Tuple): _description_
        train_loader (DataLoader): _description_
        val_loader (DataLoader): _description_
        epochs (int): _description_
        device (torch.device): _description_
        grad_clip (int): _description_
        decay_lr (int): _description_
        cfg (Dict): _description_
        wandb_log (bool, optional): _description_. Defaults to True.
        iter_num (int, optional): _description_. Defaults to 0.
        best_val_loss (int, optional): _description_. Defaults to 1e9.
    """
    optimizer1, optimizer2 = optimizers[0], optimizers[1]
    scheduler1, scheduler2 = schedulers[0], schedulers[1]
    for epoch in range(cfg["EPOCHS"]):
        model.train()
        for X, y, _ in tqdm(train_loader):
            drug, molecular, outcome = (
                X["drug_encoding"].to(cfg["DEVICE"]),
                X["gene_expression"].to(cfg["DEVICE"]),
                y.to(cfg["DEVICE"]),
            )

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            prediction = model.forward(drug, molecular)
            loss = loss_func(outcome, prediction)

            loss.backward()
            clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP"])
            # Randomly select optimizer
            selected_optimizer = optimizer1 if torch.rand(1) < 0.5 else optimizer2
            selected_optimizer.step()
            if cfg["DECAY_LR"]:
                scheduler1.step()
                scheduler2.step()
            if (iter_num % cfg["EVAL_INTERVAL"] == 0) & (iter_num != 0):
                avg_val_loss, r2_val = evaluate_model(model, val_loader, data, epoch, cfg, loss_func)
                if cfg["WANDB_LOG"]:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "train loss": loss.item(),
                            "val_loss": avg_val_loss,
                            "r2_val": r2_val,
                            "lr_opt1": optimizer1.param_groups[0]["lr"],
                            "lr_opt2": optimizer2.param_groups[0]["lr"],
                        }
                    )
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
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
                        f"New best val_loss achieved. Saving checkpoint to "
                        f"{os.path.join(cfg['RESULTS_PATH'], 'ckpt.pt')}"
                    )
                    torch.save(checkpoint, os.path.join(cfg["RESULTS_PATH"], "ckpt.pt"))
            iter_num += 1


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
