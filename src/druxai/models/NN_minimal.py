"""LRP and Interaction NN models."""

import torch.nn as nn


class Model(nn.Module):
    """_summary_.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        """_summary_."""
        # Initialize weights using Kaiming Uniform initialization
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="leaky_relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="leaky_relu")

    def forward(self, x):
        """summary.

        Args:
            x (_type_): _description_

        Returns
        -------
            _type_: _description_
        """
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        return self.fc2(x)


class Interaction_Model(nn.Module):
    """_summary_.

    Args:
        nn (_type_): _description_

    Returns
    -------
        _type_: _description_
    """

    classname = "Interaction Model"

    def __init__(self, ds):
        super().__init__()

        self.nfeatures1, self.nfeatures2, self.nfeatures_product = (ds.ndrug_features, ds.nmolecular_features, 1000)

        self.nn1 = Model(self.nfeatures1, self.nfeatures_product, hidden_dim=5000, dropout=0.05)
        self.nn2 = Model(self.nfeatures2, self.nfeatures_product, hidden_dim=10000, dropout=0.05)

    def forward(self, drug, molecular):
        """_summary_.

        Args:
            drug (_type_): _description_
            molecular (_type_): _description_

        Returns
        -------
            _type_: _description_
        """
        intermediate1 = self.nn1(drug)
        intermediate2 = self.nn2(molecular)

        product = intermediate1 * intermediate2

        return product.mean(axis=1).unsqueeze(1)
