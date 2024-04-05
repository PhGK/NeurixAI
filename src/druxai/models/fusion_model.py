"""Fusion Model end-to-end."""

import torch
from torch import nn

from druxai.utils.set_seeds import set_seeds

set_seeds()


class FusionModel(nn.Module):
    """
    A model that fuses the outputs of two separate models processing drug and gene expression data.

    Attributes
    ----------
        drug_model: A sequential model processing drug data.
        gene_expression_model: A sequential model processing gene expression data.
        fusion_layer: A linear layer that fuses the outputs of the two models.
    """

    def __init__(
        self,
        dataset,
        hidden_size: int,
        output_size: int,
        final_output_size: int,
        dropout_rate_drug_model: float = 0,
        dropout_rate_gene_expression_model: float = 0,
    ):
        """
        Initialize the FusionModel.

        Args:
            dataset: The dataset object containing the drug and gene expression data.
            hidden_size (int): The size of the hidden layers in the models.
            output_size (int): The size of the output from the models before fusion.
            final_output_size (int): The size of the final output after fusion.
            dropout_rate_drug_model (float, optional): The dropout rate for the drug model layers. Defaults to 0.
            dropout_rate_gene_expression_model (float, optional): The dropout rate for the gene expression model layers.
                                                                  Defaults to 0.
        """
        super().__init__()
        self.ndrug_features = dataset.ndrug_features
        self.nmolecular_features = dataset.nmolecular_features

        self.drug_model = nn.Sequential(
            nn.Linear(self.ndrug_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate_drug_model),
            nn.Linear(hidden_size, output_size),
        )
        self.gene_expression_model = nn.Sequential(
            nn.Linear(self.nmolecular_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate_gene_expression_model),
            nn.Linear(hidden_size, output_size),
        )
        self.fusion_layer = nn.Linear(output_size * 2, final_output_size)

    def forward(self, drug_data: torch.Tensor, gene_expression_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            drug_data (torch.Tensor): The drug data.
            gene_expression_data (torch.Tensor): The gene expression data.

        Returns
        -------
            torch.Tensor: The output of the model after fusion.
        """
        drug_output = self.drug_model(drug_data)
        gene_expression_output = self.gene_expression_model(gene_expression_data)
        fused_output = torch.cat((drug_output, gene_expression_output), dim=1)
        return self.fusion_layer(fused_output)
