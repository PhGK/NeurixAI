"""Training script. To run it do: $ python train.py --batch_size=32."""

import os

import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import pandas as pd

from druxai._logging import logger
from druxai.models.NN import Interaction_Model
from druxai.utils.data import MyDataSet

# -----------------------------------------------------------------------------
# training params
batch_size = 512
learning_rate = 0.001
fold = 1
epochs = 3
train_on_subset = False
subset_size = 5000
seed = 1337
grad_clip = 1.0
device = torch.device("mps")  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps' etc.
PATH = "/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/results/training"
init_from = "scratch"
# wandb logging
wandb_log = True
wandb_project_name = "druxai"
wandb_run_name = "druxai-run1"
# added to not get AttributeError: module '__main__' has no attribute '__spec__'
__spec__ = None
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
with open("/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/src/druxai/utils/configurator.py") as f:
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


def evaluate_model(model, ds, dataloader_val, fold, epoch, PATH, epochs):
    """Evaluate Model after each epoch of training and saves final prediction for last epoch.

    Args:
        model (_type_): _description_
        ds (_type_): _description_
        dataloader_val (_type_): Dataloader which is being evaluated.
        fold (_type_): _description_
        epoch (_type_): _description_
        PATH (_type_): _description_

    Returns
    -------
        total_loss: average test loss for batch
        r_score: Pearson correlation coefficient (r score) for the predictions
    """
    model.eval()
    prediction_frames = []
    all_outcomes = []
    all_predictions = []
    val_total_loss = 0.0

    for drug, molecular, outcome, _ in tqdm(dataloader_val):
        drug, molecular, outcome = drug.to(device), molecular.to(device), outcome.to(device)

        with torch.no_grad():
            prediction = model(drug, molecular)
            loss = nn.HuberLoss()(outcome, prediction)
            val_total_loss += loss.item()

        # TODO: bad dependency, make criterion instead of hardcoding huber loss
        loss = nn.HuberLoss()(outcome, prediction)

        all_outcomes.extend(outcome.cpu().detach().numpy().reshape(-1))
        all_predictions.extend(prediction.cpu().detach().numpy().reshape(-1))
        if epoch == epochs:
            pd.concat(prediction_frames, axis=0).to_csv(os.path.join(PATH, "prediction.csv"))

    r_score, _ = pearsonr(all_outcomes, all_predictions)
    avg_loss = val_total_loss / len(dataloader_val)
    return avg_loss, r_score


def train():
    """End-to-End Training of Druxai Network."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Some Initalizations
    best_val_loss = 1e5

    # Hard coded 2 datasets for now, should be improved in the future; Badly coded; for sure needs to be refactored
    # Works because splits are the same because of same seed
    train_data = MyDataSet(file_path="../../data/preprocessed", results_dir="../../results", n_splits=5)
    dataset_ids = train_data.generate_train_val_test_ids(seed=seed)
    train_data.preprocess_train_val_test(dataset_ids)

    val_data = MyDataSet(file_path="../../data/preprocessed", results_dir="../../results", n_splits=5)
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
        logger.info(f"Resuming training from {PATH}/ckpt.pt")
        # Load checkpoint
        ckpt_path = os.path.join(PATH, "ckpt.pt")
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

    criterion = nn.HuberLoss()

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
        r2_batch = 0.0

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

            # Calculate Pearson correlation coefficient for the current batch; Computationally not very efficient
            r2_batch, _ = pearsonr(
                outcome.cpu().detach().numpy().reshape(-1), prediction.cpu().detach().numpy().reshape(-1)
            )
            r2_batch += r2_batch

        avg_r2 = r2_batch / len(train_loader)
        avg_loss = total_loss / len(train_loader)

        avg_val_loss, r2_val = evaluate_model(model, train_data, val_loader, fold, epoch, PATH, epochs)

        if wandb_log:
            wandb.log({"train loss": avg_loss, "r2_train": avg_r2, "val_loss": avg_val_loss, "r2_val": r2_val})

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
            logger.info(f"New best val_loss achieved. Saving checkpoint to {PATH}/ckpt.pt")
            torch.save(checkpoint, os.path.join(PATH, "ckpt.pt"))

        scheduler1.step()
        scheduler2.step()

    # Finish the wandb run
    if wandb_log:
        wandb.finish()


if __name__ == "__main__":
    train()
