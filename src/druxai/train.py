"""Training script. To run it do: $ python train.py --batch_size=32."""

import os
import random
from typing import Tuple

import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd

from druxai._logging import _setup_logger
from druxai.models.NN_minimal import Interaction_Model
from druxai.utils.data import DrugResponseDataset
from druxai.utils.dataframe_utils import (
    create_batch_result,
    split_data_by_cell_line_ids,
    standardize_molecular_data_inplace,
)

# -----------------------------------------------------------------------------
# Path which need to be defined SET THESE PARAMETERS
RESULTS_PATH = "/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/results/training"
PROJECT_PATH = "/Users/niklaskiermeyer/Desktop/Codespace/DruxAI"
DATA_PATH = "/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/data/preprocessed"
batch_size = 128
epochs = 15
train_on_subset = False
subset_size = 100000
seed = 1337
grad_clip = 1.0
device = torch.device("mps")  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps' etc.
criterion = nn.HuberLoss()
num_workers = 1
init_from = "scratch"  # Use "resume" from os.path.join(RESULTS_PATH, "ckpt.pt") for pretrained weights
logger = _setup_logger(log_file_path=os.path.join(RESULTS_PATH, "logfile.log"))  # Logging config
# optimizer params
learning_rate = 0.05
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# wandb logging
wandb_log = True
wandb_project_name = "druxai"
wandb_run_name = "druxai-run1"
# -----------------------------------------------------------------------------
# Fixed params
# added to not get AttributeError: module '__main__' has no attribute '__spec__'
__spec__ = None
# -----------------------------------------------------------------------------


config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
with open(os.path.join(PROJECT_PATH, "src", "druxai", "utils", "configurator.py")) as f:
    exec(f.read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

if wandb_log:
    import wandb

    wandb.init(
        project=wandb_project_name,
        name=wandb_run_name,
        config=config,
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader_val: DataLoader,
    data: DrugResponseDataset,
    epoch: int,
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
    val_total_loss = 0.0

    for X, y, idx in tqdm(dataloader_val):
        drug, molecular, outcome = (
            X["drug_encoding"].to(torch.device("mps")),
            X["gene_expression"].to(torch.device("mps")),
            y.to(torch.device("mps")),
        )

        prediction = model(drug, molecular)
        loss = criterion(outcome, prediction)
        val_total_loss += loss.item()

        all_outcomes.extend(outcome.cpu().detach().numpy().reshape(-1))
        all_predictions.extend(prediction.cpu().detach().numpy().reshape(-1))
        if epoch == (epochs - 1):
            batch_result = create_batch_result(outcome, prediction, data, idx, epoch)
            prediction_frames.append(batch_result)

    if save_evaluation & (epoch == (epochs - 1)):
        pd.concat(prediction_frames, axis=0).to_csv(os.path.join(RESULTS_PATH, "prediction.csv"))

    if eval_metric == "spearmanr":
        r_score, _ = spearmanr(all_outcomes, all_predictions)
    elif eval_metric == "pearsonr":
        r_score, _ = pearsonr(all_outcomes, all_predictions)
    else:
        raise ValueError("Invalid evaluation metric. Use 'spearmanr' or 'pearsonr'.")

    avg_loss = val_total_loss / len(dataloader_val)
    model.train()
    return avg_loss, r_score


def train():
    """End-to-End Training of Druxai Network."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    # Some Initalizations
    best_val_loss = 1e9

    # Load data
    data = DrugResponseDataset(DATA_PATH)
    train_id, val_id, test_id = split_data_by_cell_line_ids(data.targets, seed=seed)
    standardize_molecular_data_inplace(data, train_id, val_id, test_id)
    if train_on_subset:
        train_loader = DataLoader(
            data,
            sampler=random.sample(train_id, subset_size),
            batch_size=8,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            data,
            sampler=random.sample(val_id, int(0.2 * subset_size)),
            batch_size=8,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            data, sampler=train_id, batch_size=8, shuffle=False, pin_memory=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            data, sampler=val_id, batch_size=8, shuffle=False, pin_memory=True, num_workers=num_workers
        )

    # Determine whether to initialize a new model from scratch or resume training from a checkpoint
    if init_from == "scratch":
        logger.info("Training model from scratch.")
        model = Interaction_Model(data)
        model.train().to(device)
        # Setup optimizers
        optimizer1 = AdamW(model.nn1.parameters(), learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        optimizer2 = AdamW(model.nn2.parameters(), learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    elif init_from == "resume":
        logger.info(f"Resuming training from {RESULTS_PATH}/ckpt.pt")
        # Load checkpoint
        ckpt_path = os.path.join(RESULTS_PATH, "ckpt.pt")
        checkpoint = torch.load(ckpt_path)
        # Initialize model with the same configuration as the checkpoint
        model = Interaction_Model(data)
        model.train().to(device)
        model.load_state_dict(checkpoint["model"])
        # Setup optimizers
        optimizer1 = AdamW(model.nn1.parameters(), learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        optimizer2 = AdamW(model.nn2.parameters(), learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        # Also load optimizer states if needed
        optimizer1.load_state_dict(checkpoint["optimizer 1"])
        optimizer2.load_state_dict(checkpoint["optimizer 2"])
        # Update best validation loss
        best_val_loss = checkpoint["best_val_loss"]
        model.train().to(device)
    else:
        raise ValueError("init_from must be either 'scratch' or 'resume'")

    scheduler1 = ExponentialLR(optimizer1, gamma=0.9)
    scheduler2 = ExponentialLR(optimizer2, gamma=0.9)

    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X, y, _ in tqdm(train_loader):
            drug, molecular, outcome = (
                X["drug_encoding"].to(torch.device("mps")),
                X["gene_expression"].to(torch.device("mps")),
                y.to(torch.device("mps")),
            )

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            prediction = model.forward(drug, molecular)
            loss = criterion(outcome, prediction)
            total_loss += loss.item()

            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            # Randomly select optimizer
            selected_optimizer = optimizer1 if torch.rand(1) < 0.5 else optimizer2
            selected_optimizer.step()

        avg_loss = total_loss / len(train_loader)

        avg_val_loss, r2_val = evaluate_model(model, val_loader, data, epoch)

        if wandb_log:
            wandb.log(
                {
                    "train loss": avg_loss,
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
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            logger.info(f"New best val_loss achieved. Saving checkpoint to {os.path.join(RESULTS_PATH, 'ckpt.pt')}")
            torch.save(checkpoint, os.path.join(RESULTS_PATH, "ckpt.pt"))

        scheduler1.step()
        scheduler2.step()

    # Finish the wandb run
    if wandb_log:
        wandb.finish()


if __name__ == "__main__":
    train()
