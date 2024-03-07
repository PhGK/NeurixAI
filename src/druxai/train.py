"""Training Model to train NN."""

import os

import torch
import torch.nn as nn
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import pandas as pd

from druxai.utils.dataframe_utils import create_batch_result


def train_model(model, ds, lr, nepochs, fold, device, PATH="../../results/training/", wandb_log=False, on_subset=False):
    """Train model.

    Args:
        model (_type_): _description_
        ds (_type_): Dataset
        lr (_type_): _description_
        nepochs (_type_): _description_
        fold (_type_): _description_
        device (_type_): _description_
        PATH (str, optional): _description_. Defaults to "../../results/training/".
        wandb_log (bool, optional): _description_. Defaults to False.
        on_subset (bool, optional): Whether to train on subset for fast testing. Defaults to False.

    Returns
    -------
        _type_: _description_
    """
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Initialize wandb run
    if wandb_log:
        wandb.init(
            project="druxai",
            config={
                "learning_rate": lr,
                "epochs": nepochs,
            },
        )

    model.train().to(device)

    optimizer1 = SGD(model.nn1.parameters(), momentum=0.9, lr=lr, weight_decay=1e-5)
    optimizer2 = SGD(model.nn2.parameters(), momentum=0.9, lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss()

    scheduler1 = ExponentialLR(optimizer1, gamma=0.9)  # gamma 0.9 #0.95 only for 100 epochs
    scheduler2 = ExponentialLR(optimizer2, gamma=0.9)  # gamma 0.9

    ds.change_fold(fold, "train")
    for epoch in range(1, nepochs + 1):  # been without +1
        bs = 128

        # Small subset for testing
        if on_subset:
            subset_indices = range(0, 5000)
            subset_dataset = Subset(ds, subset_indices)
            dl = DataLoader(subset_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)

        else:
            dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)

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
            clip_grad_norm_(model.parameters(), 1.0)

            # Randomly select optimizer
            selected_optimizer = optimizer1 if torch.rand(1) < 0.5 else optimizer2
            selected_optimizer.step()

        ds.change_fold(fold, "test")
        assert ds.mode == "test", "testing on " + ds.mode + " dataset is not possible"

        dl_test = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
        avg_test_loss = evaluate_model(model, ds, dl_test, device, fold, epoch, PATH, nepochs)

        if wandb_log:
            wandb.log({"epoch": epoch, "epoch loss": train_loss, "test_loss": avg_test_loss})

        ds.change_fold(fold, "train")

        scheduler1.step()
        scheduler2.step()

    # Finish the wandb run
    if wandb_log:
        wandb.finish()

    return model


@torch.no_grad()
def evaluate_model(model, ds, dl_test, device, fold, epoch, PATH, nepochs):
    """Evaluate Model after each epoch of training and saves final prediction for last epoch.

    Args:
        model (_type_): _description_
        ds (_type_): _description_
        dl_test (_type_): _description_
        device (_type_): _description_
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

        if epoch == nepochs:
            pd.concat(prediction_frames, axis=0).to_csv(os.path.join(PATH, "prediction.csv"))

    return avg_test_loss
