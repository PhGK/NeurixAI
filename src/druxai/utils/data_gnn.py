"""Drug Response Data Module."""

import os
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import pandas as pd

from druxai._logging import logger
from druxai.utils.gnn_utils import load_dictionary
from druxai.utils.set_seeds import set_seeds

set_seeds()


class DrugResponseDataset(Dataset):
    """Drug Response Dataset class which handles molecular and drug data."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.targets = pd.read_csv(
            os.path.join(file_path, "auc_secondary_screen_prediction_targets.csv"), index_col=0
        ).reset_index(drop=True)
        logger.info(f"Loaded targets with shape: {self.targets.shape}")
        self.molecular_data = pd.read_csv(os.path.join(file_path, "rna_df.csv"), index_col=0)
        logger.info(f"Loaded molecular data with shape: {self.molecular_data.shape}")
        # Check for NaNs and duplicates
        assert not self.targets.isnull().values.any(), "NaN values detected in self.targets DataFrame."
        assert not self.molecular_data.isnull().values.any(), "NaN values detected in self.molecular_data DataFrame."
        assert not self.targets.duplicated().any(), "Duplicates detected in self.targets DataFrame."
        assert not self.molecular_data.duplicated().any(), "Duplicates detected in self.molecular_data DataFrame."

        self.smile_code_dict = load_dictionary(os.path.join(file_path, "drug_gnn_embeddings.pt"))

        # in future change NN class directly so no need to specify here
        self.ndrug_features = len(self.targets["DRUG"].unique())
        self.nmolecular_features = self.molecular_data.shape[1]

    def __getitem__(self, idx):
        """Return Dict containing gene expression and drug encoding and target as well as id."""
        gene_expression_values = torch.tensor(
            self.molecular_data.loc[self.targets.iloc[idx]["cell_line"]].values, dtype=torch.float32
        )
        target = torch.tensor([self.targets.iloc[idx]["auc_per_drug"]], dtype=torch.float32)

        drug_smile_enc = self.smile_code_dict[self.targets.iloc[idx]["DRUG"]]

        return gene_expression_values, drug_smile_enc, target, idx

    def __len__(self):
        """Return length of the targets."""
        return len(self.targets)


class DataloaderSampler(Sampler):
    """Sampler that generates indices based on a fixed list of indices."""

    def __init__(self, indices):
        """
        Initialize the FixedSampler with a list of fixed indices.

        Args:
            indices (list): A list of fixed indices.
        """
        self.indices = indices

    def __iter__(self):
        """
        Return an iterator over the fixed indices. Shuffles the indices at the beginning of each epoch.

        Returns
        -------
            iterator: An iterator over the fixed indices.
        """
        random.shuffle(self.indices)  # Shuffle indices at the beginning of each epoch
        return iter(self.indices)

    def __len__(self):
        """
        Return the number of fixed indices.

        Returns
        -------
            int: The number of fixed indices.
        """
        return len(self.indices)
