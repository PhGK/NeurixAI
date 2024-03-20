"""Training script. To run it do: $ python train.py --batch_size=32."""

import os
from typing import Tuple

import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import pandas as pd

from druxai._logging import _setup_logger
from druxai.models.NN import Interaction_Model
from druxai.utils.data import MyDataSet
from druxai.utils.dataframe_utils import create_batch_result

# -----------------------------------------------------------------------------
# path which need to be defined
RESULTS_PATH = "/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/results/training"
PROJECT_PATH = "/Users/niklaskiermeyer/Desktop/Codespace/DruxAI"
DATA_PATH = "/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/data/preprocessed"
# NN training params
batch_size = 512
learning_rate = 0.001
fold = 1
epochs = 3
train_on_subset = False
subset_size = 5000
seed = 1337
grad_clip = 1.0
device = torch.device("mps")  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps' etc.
criterion = nn.HuberLoss()
# Use Preloaded weights "resume" from os.path.join(RESULTS_PATH, "ckpt.pt") or start from scratch
init_from = "scratch"
# Logging config
logger = _setup_logger(log_file_path=os.path.join(RESULTS_PATH, "logfile.log"))
# wandb logging
wandb_log = True
wandb_project_name = "druxai"
wandb_run_name = "druxai-run1"
# added to not get AttributeError: module '__main__' has no attribute '__spec__'
__spec__ = None
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
with open(os.path.join(PROJECT_PATH, "src", "druxai", "utils", "configurator.py")) as f:
    exec(f.read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# Initialize wandb run if wandb_log is True
if wandb_log:
    import wandb

    wandb.init(
        project=wandb_project_name,
        name=wandb_run_name,
        config=config,
    )


def evaluate_model(
    model: torch.nn.Module, dataloader_val: DataLoader, val_data: MyDataSet, epoch: int, save_evaluation: bool = True
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

    for drug, molecular, outcome, idx in tqdm(dataloader_val):
        drug, molecular, outcome = drug.to(device), molecular.to(device), outcome.to(device)

        with torch.no_grad():
            prediction = model(drug, molecular)
            loss = criterion(outcome, prediction)
            val_total_loss += loss.item()

        all_outcomes.extend(outcome.cpu().detach().numpy().reshape(-1))
        all_predictions.extend(prediction.cpu().detach().numpy().reshape(-1))
        if epoch == (epochs - 1):
            batch_result = create_batch_result(outcome, prediction, val_data, idx, epoch)
            prediction_frames.append(batch_result)

    if save_evaluation & (epoch == (epochs - 1)):
        pd.concat(prediction_frames, axis=0).to_csv(os.path.join(RESULTS_PATH, "prediction.csv"))

    r_score, _ = pearsonr(all_outcomes, all_predictions)
    avg_loss = val_total_loss / len(dataloader_val)
    return avg_loss, r_score


def train():
    """End-to-End Training of Druxai Network."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    # Some Initalizations
    best_val_loss = 1e5

    # Hard coded 2 datasets for now, should be improved in the future; Badly coded; for sure needs to be refactored
    # Works because splits are the same because of same seed
    train_data = MyDataSet(file_path=DATA_PATH, results_dir=RESULTS_PATH, n_splits=5)
    dataset_ids = train_data.generate_train_val_test_ids(seed=seed)
    train_data.preprocess_train_val_test(dataset_ids)

    val_data = MyDataSet(file_path=DATA_PATH, results_dir=RESULTS_PATH, n_splits=5)
    dataset_ids = val_data.generate_train_val_test_ids(seed=seed)
    val_data.preprocess_train_val_test(dataset_ids)

    # Determine whether to initialize a new model from scratch or resume training from a checkpoint
    if init_from == "scratch":
        logger.info("Training model from scratch.")
        model = Interaction_Model(train_data)
        model.train().to(device)
        # Setup optimizers
        optimizer1 = SGD(model.nn1.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-5)
        optimizer2 = SGD(model.nn2.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-5)

    elif init_from == "resume":
        logger.info(f"Resuming training from {RESULTS_PATH}/ckpt.pt")
        # Load checkpoint
        ckpt_path = os.path.join(RESULTS_PATH, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Initialize model with the same configuration as the checkpoint
        model = Interaction_Model(train_data)
        model.load_state_dict(checkpoint["model"])
        # Setup optimizers
        optimizer1 = SGD(model.nn1.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-5)
        optimizer2 = SGD(model.nn2.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-5)
        # Also load optimizer states if needed
        optimizer1.load_state_dict(checkpoint["optimizer 1"])
        optimizer2.load_state_dict(checkpoint["optimizer 2"])
        # Update best validation loss
        best_val_loss = checkpoint["best_val_loss"]
        # Move model to the appropriate device
        model.to(device)
    else:
        raise ValueError("init_from must be either 'scratch' or 'resume'")

    scheduler1 = ExponentialLR(optimizer1, gamma=0.9)  # gamma 0.9 #0.95 only for 100 epochs
    scheduler2 = ExponentialLR(optimizer2, gamma=0.9)  # gamma 0.9

    # Choose to use subset for experimental purposes or entire dataset
    val_data.selected_dataset = "val"
    if train_on_subset:
        subset_indices = range(0, int(0.3 * subset_size))
        subset_dataset = Subset(val_data, subset_indices)
        val_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    else:
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=6)

    train_data.selected_dataset = "train"
    if train_on_subset:
        subset_indices = range(0, subset_size)
        subset_dataset = Subset(train_data, subset_indices)
        train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=6)

    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for drug, molecular, outcome, _ in tqdm(train_loader):
            drug, molecular, outcome = drug.to(device), molecular.to(device), outcome.to(device)

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

        avg_val_loss, r2_val = evaluate_model(model, val_loader, val_data, epoch)

        if wandb_log:
            wandb.log({"train loss": avg_loss, "val_loss": avg_val_loss, "r2_val": r2_val})

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
