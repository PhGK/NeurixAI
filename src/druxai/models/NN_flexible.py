"""LRP and Interaction NN models."""

from typing import List

import torch
import torch.nn as nn

from druxai.utils.set_seeds import set_seeds

set_seeds()


class Model(nn.Module):
    """Custom model with flexible hidden layer dimensions.

    Args:
        input_dim (int): Dimensionality of input features.
        output_dim (int): Dimensionality of output.
        hidden_dims (List[int]): List of integers representing the dimensions of hidden layers.
        dropout (float): Dropout probability.

    Attributes
    ----------
        model (nn.Sequential): Sequential model containing linear layers, activations, and dropout.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
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


class Interaction_Model(nn.Module):
    """Model combining NNs for interaction prediction between drug and molecular features."""

    def __init__(
        self,
        ds,
        nfeatures_product: int,
        hidden_dims_drug_nn: List[int],
        hidden_dims_gene_expression_nn: List[int],
        droupout_drug_nn: float,
        droupout_gene_expression_nn: float,
    ):
        """
        Initialize the Interaction_Model.

        Args:
            ds: Dataset object containing information about features.
            nfeatures_product (int): Dimensionality of the product features (Final output.)
            hidden_dims_drug_nn (List[int]): List of integers representing the dimensions of hidden layers for drug NN.
            hidden_dims_gene_expression_nn (List[int]): List of integers representing the dimensions of hidden layers
                                                        for Gene NN.
            droupout_drug_nn (float): Dropout probability for drug NN.
            droupout_gene_expression_nn (float): Dropout probability for gene expression NN.
        """
        super().__init__()

        # Assertion for drug features
        assert isinstance(ds.ndrug_features, int), "The number of drug features must be an integer"
        assert ds.ndrug_features > 0, "The number of drug features must be greater than zero"

        # Assertion for molecular features
        assert isinstance(ds.nmolecular_features, int), "The number of molecular features must be an integer"
        assert ds.nmolecular_features > 0, "The number of molecular features must be greater than zero"

        # Assertion for product features
        assert isinstance(nfeatures_product, int), "The number of product features must be an integer"
        assert nfeatures_product > 0, "The number of product features must be greater than zero"

        assert all(
            isinstance(dim, int) and dim > 0 for dim in hidden_dims_drug_nn
        ), "Invalid hidden dimensions for Drug NN"
        assert all(
            isinstance(dim, int) and dim > 0 for dim in hidden_dims_gene_expression_nn
        ), "Invalid hidden dimensions for Gene NN"
        assert 0.0 <= droupout_drug_nn < 1.0, "Dropout probability for Drug NN must be in range [0, 1)"
        assert 0.0 <= droupout_gene_expression_nn < 1.0, "Dropout probability for Gene NN must be in range [0, 1)"

        self.input_features_drug_nn = ds.ndrug_features
        self.input_features_gene_expression_nn = ds.nmolecular_features
        self.nfeatures_product = nfeatures_product
        self.hidden_dims_drug_nn = hidden_dims_drug_nn
        self.hidden_dims_gene_expression_nn = hidden_dims_gene_expression_nn
        self.droupout_drug_nn = droupout_drug_nn
        self.droupout_gene_expression_nn = droupout_gene_expression_nn

        # Define neural networks
        self.drug_nn = Model(
            self.input_features_drug_nn,
            self.nfeatures_product,
            hidden_dims=self.hidden_dims_drug_nn,
            dropout=self.droupout_drug_nn,
        )
        self.gene_expression_nn = Model(
            self.input_features_gene_expression_nn,
            self.nfeatures_product,
            hidden_dims=self.hidden_dims_gene_expression_nn,
            dropout=self.droupout_gene_expression_nn,
        )

    def forward(self, drug: torch.Tensor, molecular: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Interaction_Model.

        Args:
            drug (torch.Tensor): Tensor containing drug features.
            molecular (torch.Tensor): Tensor containing molecular features.

        Returns
        -------
            torch.Tensor: Tensor representing the interaction of drug and molecular features.
        """
        intermediate1 = self.drug_nn(drug)
        intermediate2 = self.gene_expression_nn(molecular)
        product = intermediate1 * intermediate2
        return product.mean(axis=1).unsqueeze(1)
