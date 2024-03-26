"""Drug Response Data Module."""

import os
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import pandas as pd

from druxai._logging import logger

# from druxai._logging import logger


class DrugResponseDataset(Dataset):
    """Drug Response Dataset class which handels molecular and drug data."""

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

        self.drug_encoding_dict = {
            drug: torch.zeros(len(self.targets["DRUG"].unique())).scatter_(0, torch.tensor(i), 1)
            for i, drug in enumerate(self.targets["DRUG"].unique())
        }
        # in future change NN class directly so no need to specify here
        self.ndrug_features = len(self.targets["DRUG"].unique())
        self.nmolecular_features = self.molecular_data.shape[1]

    def __getitem__(self, idx):
        """Return Dict containing gene expression and drug encoding and target as well as id."""
        gene_expression_values = torch.tensor(
            self.molecular_data.loc[self.targets.iloc[idx]["cell_line"]].values, dtype=torch.float32
        )
        target = torch.tensor([self.targets.iloc[idx]["logauc"]], dtype=torch.float32)
        drug_one_hot = self.drug_encoding_dict[self.targets.iloc[idx]["DRUG"]]
        return {"gene_expression": gene_expression_values, "drug_encoding": drug_one_hot}, target, idx

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
