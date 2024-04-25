"""GNN model architecture."""

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

from druxai.utils.set_seeds import set_seeds

set_seeds()


class Drug_GNN(torch.nn.Module):
    """
    A Graph Neural Network model for drug data.

    This model applies a Graph Convolutional Network (GCN) to the input data,
    followed by a global mean pooling layer and a fully connected layer.

    Attributes
    ----------
        output_size (int): The size of the output tensor.
        conv1 (GCNConv): The first graph convolutional layer.
        fc (Linear): The fully connected layer.
    """

    def __init__(self, output_size: int = 1):
        """
        Initialize the Drug_GNN.

        Args:
            output_size (int, optional): The size of the output tensor. Defaults to 1.
        """
        super().__init__()
        self.output_size = output_size
        self.conv1 = GCNConv(79, 64)
        self.fc = Linear(64, output_size)

    def forward(self, data: Batch) -> Tensor:
        """
        Forward pass through the network.

        Args:
            data (Batch): The input data, a batch of graphs.

        Returns
        -------
            Tensor: The output of the network.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1st Graph Convolution
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # Global Pooling (mean)
        x = global_mean_pool(x, batch)

        return self.fc(x)


class Drug_GAT(torch.nn.Module):
    """
    A Graph Attention Network (GAT) model for drug data.

    This model applies a Graph Attention Network (GAT) to the input data,
    followed by a global mean pooling layer and a fully connected layer.

    Attributes
    ----------
        output_size (int): The size of the output tensor.
        conv1 (GATConv): The first graph attention layer.
        fc (Linear): The fully connected layer.
    """

    def __init__(self, output_size: int = 1):
        """
        Initialize the Drug_GAT.

        Args:
            output_size (int, optional): The size of the output tensor. Defaults to 1.
        """
        super().__init__()
        self.output_size = output_size
        self.conv1 = GATConv(79, 128, heads=8, dropout=0.5)  # Adjust parameters as needed
        self.fc = Linear(128 * 8, output_size)  # Adjust input size for fully connected layer

    def forward(self, data: Batch) -> Tensor:
        """
        Forward pass through the network.

        Args:
            data (Batch): The input data, a batch of graphs.

        Returns
        -------
            Tensor: The output of the network.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1st Graph Attention Layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # Global Pooling (mean)
        x = global_mean_pool(x, batch)

        return self.fc(x)
