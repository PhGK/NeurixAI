"""Module which creates NN model for single modality."""

from typing import List

import torch
import torch.nn as nn


class Single_Model(nn.Module):
    """Single Model for training to check against combined performance of gene and drug models.

    Args:
        input_dim (int): Dimensionality of input features.
        hidden_dims (List[int]): List of integers representing the dimensions of hidden layers.
        dropout (float): Dropout probability.

    Attributes
    ----------
        model (nn.Sequential): Sequential model containing linear layers, activations, and dropout.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Output tensor.
        """
        return self.model(x)
