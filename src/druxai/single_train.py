"""Train a single model on data - either gene expression as input or drug encoding."""

from types import SimpleNamespace

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

import wandb
from druxai.models.single_model import Single_Model
from druxai.utils.data import DrugResponseDataset
from druxai.utils.dataframe_utils import (
    split_data_by_cell_line_ids,
    standardize_molecular_data_inplace,
)

## Configs ##
#############


def main():
    """Train a single model on data."""
    config = {
        "file_path": "/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/data/preprocessed",
        "input_feature": "gene_expression",  # or "gene_expression" or "drug_encoding"
        "epochs": 25,
        "input_features": 19193,  # for genes: 19193, for drugs: 1149
        "hidden_dims": [100],
        "name": "gene_network_train",
        "dropout": 0.2,
    }
    #############

    config = SimpleNamespace(**config)

    # Initialize a new wandb run
    wandb.init(project="DruxAI Hyperparameter Sweep", name=config.name, config=config)

    data = DrugResponseDataset(config.file_path)

    train_id, val_id, test_id = split_data_by_cell_line_ids(data.targets)

    standardize_molecular_data_inplace(data, train_id=train_id, val_id=val_id, test_id=test_id)

    # Dataloader
    train_loader = DataLoader(data, sampler=train_id, batch_size=128, shuffle=False, pin_memory=True, num_workers=6)
    val_loader = DataLoader(data, sampler=val_id, batch_size=128, shuffle=False, pin_memory=True, num_workers=6)

    # Train Loop
    model = Single_Model(config.input_features, config.hidden_dims, config.dropout)
    model.train().to(torch.device("mps"))

    # Setup optimizers
    optimizer = SGD(model.parameters(), momentum=0.9, lr=0.01, weight_decay=1e-5)

    epoch = 0
    while epoch < config.epochs:
        model.train()
        total_loss = 0.0
        train_predictions, train_outcomes = [], []

        for X, y, _ in tqdm(train_loader):
            input = X[config.input_feature].to(torch.device("mps"))
            outcome = y.to(torch.device("mps"))
            optimizer.zero_grad()

            prediction = model.forward(input)
            train_predictions.extend(prediction.detach().cpu().numpy())
            train_outcomes.extend(outcome.cpu().numpy())

            loss = nn.HuberLoss()(outcome, prediction)
            total_loss += loss.item()

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Calculate the Spearman's rank correlation coefficient for training set
        train_spearman_r = spearmanr(train_predictions, train_outcomes)[0]
        wandb.log({"total_loss": np.mean(total_loss), "train_spearman_r": train_spearman_r})

        # Evaluation loop on the validation set
        model.eval()
        val_predictions, val_outcomes, val_losses = [], [], []
        with torch.no_grad():
            for X, y, _ in val_loader:
                input = X[config.input_feature].to(torch.device("mps"))
                outcome = y.to(torch.device("mps"))
                prediction = model.forward(input)
                val_predictions.extend(prediction.cpu().numpy())
                val_outcomes.extend(outcome.cpu().numpy())
                val_loss = nn.HuberLoss()(outcome, prediction)
                val_losses.append(val_loss.item())

        val_spearman_r = spearmanr(val_predictions, val_outcomes)[0]
        avg_val_loss = np.mean(val_losses)
        wandb.log({"avg_val_loss": avg_val_loss, "val_spearman_r": val_spearman_r})

        epoch += 1

    wandb.finish()


if __name__ == "__main__":
    main()
