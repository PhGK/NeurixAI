"""Training Script for GNN Interaction Model with SMILES Graph and Gene Expression Data."""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

from druxai.models.NN_flexible import GNN_Interaction_Model
from druxai.utils.data_gnn import DataloaderSampler, DrugResponseDataset
from druxai.utils.dataframe_utils import (
    split_data_by_cell_line_ids,
    standardize_molecular_data_inplace,
)
from druxai.utils.gnn_utils import custom_collate

file_path = "/Users/niklaskiermeyer/Desktop/Codespace/DruxAI/data/preprocessed"

# Load Data
data = DrugResponseDataset(file_path)
train_id, val_id, test_id = split_data_by_cell_line_ids(data.targets)
standardize_molecular_data_inplace(data, train_id=train_id, val_id=val_id, test_id=test_id)

# Dataloader Sampler
train_sampler = DataloaderSampler(train_id)
val_sampler = DataloaderSampler(val_id)

# Dataloader
train_loader = DataLoader(data, sampler=train_sampler, batch_size=256, shuffle=False, collate_fn=custom_collate)
val_loader = DataLoader(data, sampler=val_sampler, batch_size=256, shuffle=False, collate_fn=custom_collate)

# Setup Model
model = GNN_Interaction_Model(
    data, nfeatures_product=10, hidden_dims_gene_expression_nn=[32], dropout_gene_expression_nn=0.2
)

# Setup Optimizers
optimizer1 = Adam(model.drug_gnn.parameters(), lr=0.08, weight_decay=1e-5)
optimizer2 = Adam(model.gene_expression_nn.parameters(), lr=0.08, weight_decay=1e-5)

# Training Loop
epoch = 0
while epoch < 15:
    model.train()
    for X in train_loader:
        molecular = X["gene_expression_values"].to(torch.device("cpu"))
        smile_graphs = X["smile_graph"].to(torch.device("cpu"))
        outcome = X["target"].to(torch.device("cpu"))
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        prediction = model.forward(smile_graphs, molecular)
        loss = nn.HuberLoss()(prediction, outcome)

        loss.backward()

        clip_grad_norm_(model.parameters(), 1.0)

        optimizer1.step()
        optimizer2.step()

    epoch += 1
