"""Training script. To run it do: $ python train.py --batch_size=32."""

import os

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import pandas as pd

from druxai.models.NN import Interaction_Model
from druxai.utils.data import MyDataSet
from druxai.utils.dataframe_utils import create_batch_result

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
# wandb logging
wandb_log = True
wandb_project_name = "druxai"
wandb_run_name = "druxai-run0"
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


@torch.no_grad()
def evaluate_model(model, ds, dl_test, fold, epoch, PATH, epochs):
    """Evaluate Model after each epoch of training and saves final prediction for last epoch.

    Args:
        model (_type_): _description_
        ds (_type_): _description_
        dl_test (_type_): _description_
        fold (_type_): _description_
        epoch (_type_): _description_
        PATH (_type_): _description_

    Returns
    -------
        avg_test_loss: average test loss for batch
    """
    model.eval()
    avg_test_loss = 0
    prediction_frames = []

    for drug, molecular, outcome, idx in tqdm(dl_test):
        drug, molecular, outcome = drug.to(device), molecular.to(device), outcome.to(device)
        prediction = model(drug, molecular)
        loss = nn.HuberLoss()(outcome, prediction)

        batch_result = create_batch_result(outcome, prediction, ds, idx, fold, epoch)
        prediction_frames.append(batch_result)
        avg_test_loss += loss.item() / len(dl_test)

        if epoch == epochs:
            pd.concat(prediction_frames, axis=0).to_csv(os.path.join(PATH, "prediction.csv"))

    return avg_test_loss


def train():
    """End-to-End Training of Druxai Network."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Hard coded for now, should be improved in the future
    ds = MyDataSet(file_path="../../data/preprocessed", results_dir="../../results", n_splits=5)
    model = Interaction_Model(ds)
    model.train().to(device)

    optimizer1 = SGD(model.nn1.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-5)
    optimizer2 = SGD(model.nn2.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-5)
    criterion = nn.HuberLoss()

    scheduler1 = ExponentialLR(optimizer1, gamma=0.9)  # gamma 0.9 #0.95 only for 100 epochs
    scheduler2 = ExponentialLR(optimizer2, gamma=0.9)  # gamma 0.9

    ds.change_fold(fold, "train")

    # Small subset for testing
    if train_on_subset:
        subset_indices = range(0, subset_size)
        subset_dataset = Subset(ds, subset_indices)
        dl = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    for epoch in range(1, epochs + 1):  # been without +1
        model.train()
        train_loss = 0.0
        for drug, molecular, outcome, _ in tqdm(dl):
            assert ds.mode == "train", "training on " + ds.mode + " dataset is not possible"

            drug, molecular, outcome = drug.to(device), molecular.to(device), outcome.to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            prediction = model.forward(drug, molecular)
            loss = criterion(outcome, prediction)

            # Log the loss
            if wandb_log:
                wandb.log({"Iteration Loss": loss.item()})

            # Batch train loss
            train_loss += loss.item() * drug.size(0)

            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)

            # Randomly select optimizer
            selected_optimizer = optimizer1 if torch.rand(1) < 0.5 else optimizer2
            selected_optimizer.step()

        ds.change_fold(fold, "test")
        assert ds.mode == "test", "testing on " + ds.mode + " dataset is not possible"

        dl_test = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        avg_test_loss = evaluate_model(model, ds, dl_test, fold, epoch, PATH, epochs)

        if wandb_log:
            wandb.log({"epoch": epoch, "epoch loss": train_loss, "test_loss": avg_test_loss})

        ds.change_fold(fold, "train")

        scheduler1.step()
        scheduler2.step()

    # Finish the wandb run
    if wandb_log:
        wandb.finish()


if __name__ == "__main__":
    train()
