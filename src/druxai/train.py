"""Training script. To run it do: $ python train.py --batch_size=32."""

import math
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
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
epochs = 30
train_on_subset = False
subset_size = 100000
seed = 1337
grad_clip = 1.0
device = torch.device("mps")  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps' etc.
criterion = nn.HuberLoss()
num_workers = 6
init_from = "scratch"  # Use "resume" from os.path.join(RESULTS_PATH, "ckpt.pt") for pretrained weights
logger = _setup_logger(log_file_path=os.path.join(RESULTS_PATH, "logfile.log"))  # Logging config
# eval settings
eval_interval = 100  # after how many batches we evaluate, and how many train_loss uses to approximate
# learning rate decay settings
learning_rate = 0.05
decay_lr = True  # whether to decay the learning rate
warmup_iters = 200  # how many steps to warm up for
lr_decay_iters = 1000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# optimizer params
optimizer = "AdamW"  # this is only for wandb logging; to set optimizer you need to change code
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# wandb logging
wandb_log = True
wandb_project_name = "druxai"
wandb_run_name = "druxai-run-1"
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
    losses = torch.zeros(len(dataloader_val))
    for iter_count, (X, y, idx) in tqdm(enumerate(dataloader_val)):
        drug, molecular, outcome = (
            X["drug_encoding"].to(device),
            X["gene_expression"].to(device),
            y.to(device),
        )

        prediction = model(drug, molecular)
        loss = criterion(outcome, prediction)
        losses[iter_count] = loss.item()

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

    avg_loss = losses.mean()
    model.train()
    return avg_loss, r_score


def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, learning_rate: float, min_lr: float) -> float:
    """Learning rate decay scheduler with exponential decay.

    Args:
        it (int): Current iteration.
        warmup_iters (int): Number of warmup iterations.
        lr_decay_iters (int): Number of learning rate decay iterations.
        learning_rate (float): Initial learning rate.
        min_lr (float): Minimum learning rate.

    Returns
    -------
        float: Modified learning rate.
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3) in between, use exponential decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = math.exp(-decay_ratio)  # Exponential decay
    return min_lr + coeff * (learning_rate - min_lr)


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
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            data,
            sampler=random.sample(val_id, int(0.2 * subset_size)),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            data,
            sampler=train_id,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            data,
            sampler=val_id,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
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
        iter_num = checkpoint["iter_num"]
        model.train().to(device)
    else:
        raise ValueError("init_from must be either 'scratch' or 'resume'")

    # training loop
    iter_num = 0

    # Initialize a circular buffer to store losses of the last eval_interval iterations
    for epoch in range(epochs):
        model.train()

        for X, y, _ in tqdm(train_loader):

            drug, molecular, outcome = (
                X["drug_encoding"].to(device),
                X["gene_expression"].to(device),
                y.to(device),
            )

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            prediction = model.forward(drug, molecular)
            loss = criterion(outcome, prediction)
            if decay_lr:
                lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
                for param_group1, param_group2 in zip(optimizer1.param_groups, optimizer2.param_groups):
                    param_group1["lr"] = lr
                    param_group2["lr"] = lr

            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            # Randomly select optimizer
            selected_optimizer = optimizer1 if torch.rand(1) < 0.5 else optimizer2
            selected_optimizer.step()
            if (iter_num % eval_interval == 0) & (iter_num != 0):
                avg_val_loss, r2_val = evaluate_model(model, val_loader, data, epoch)
                if wandb_log:
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
                        "config": config,
                    }
                    logger.info(
                        f"New best val_loss achieved. Saving checkpoint to {os.path.join(RESULTS_PATH, 'ckpt.pt')}"
                    )
                    torch.save(checkpoint, os.path.join(RESULTS_PATH, "ckpt.pt"))
            iter_num += 1

    # Finish the wandb run
    if wandb_log:
        wandb.finish()


if __name__ == "__main__":
    train()
