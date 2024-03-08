"""Compound Data Module."""

import os
from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from druxai._logging import logger


class MyDataSet(Dataset):
    """Dataset class which contains molecular information, target auc, drug features and cell types.

    Args:
        Dataset (torch.utils.data.Dataset): Pytorch dataset object.
    """

    def __init__(self, file_path, n_splits, results_dir):
        self.file_path = file_path
        self.splits = n_splits
        self.results_dir = results_dir

        self.cases = pd.read_csv(
            os.path.join(file_path, "auc_secondary_screen_prediction_targets.csv"),
            index_col=0,
        ).reset_index(drop=True)
        self.molecular_data = pd.read_csv(os.path.join(file_path, "rna_df.csv"), index_col=0)
        self.unique_drugs, self.unique_cell_lines = (
            self.cases["DRUG"].drop_duplicates(),
            self.cases["cell_line"].drop_duplicates(),
        )
        self.unique_drugs_array = np.array(self.unique_drugs)
        self.drug_onehot_tensor = torch.eye(self.unique_drugs.shape[0])
        self.molecular_names = np.array(self.molecular_data.columns)
        self.ndrug_features = self.unique_drugs.shape[0]
        self.nmolecular_features = self.molecular_data.shape[1]

        os.makedirs(results_dir, exist_ok=True)
        pd.DataFrame({"names": self.molecular_names}).to_csv(os.path.join(results_dir, "correct_gene_order.csv"))

        self.cell_line = np.array(self.cases["cell_line"].drop_duplicates().sort_values())
        self.ncelltypes = self.cell_line.shape[0]

        self.cell_ids_test_lists, self.cell_lines_test_lists = self.generate_train_and_test_ids()

        logger.info(
            "%d molecular features, %d unique drugs, %d unique cell lines",
            self.nmolecular_features,
            self.unique_drugs.shape[0],
            self.unique_cell_lines.shape[0],
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Retrieve a sample based on the provided index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]: A tuple containing the drug features,
                molecular data, target outcome, and index of the sample.
        """
        case = self.current_cases.loc[idx, :]
        current_drugname = case["DRUG"]
        current_model_id = case["cell_line"]

        current_drug_onehot = self.drug_onehot_tensor[:, self.unique_drugs_array == current_drugname].squeeze()
        current_molecular_data = self.current_molecular_data_tensor[
            torch.tensor(self.current_molecular_data.index == current_model_id).squeeze(),
            :,
        ].squeeze()
        current_outcome = torch.tensor(case["auc_per_drug"]).float()

        return (current_drug_onehot, current_molecular_data, current_outcome.unsqueeze(0), idx)

    def __len__(self):
        """_summary_.

        Returns
        -------
            _type_: _description_
        """
        return self.current_cases.shape[0]

    def change_fold(self, fold, train_test):
        """Set fold and check if in train or test set.

        Args:
            fold (_type_): _description_
            train_test (_type_): _description_
        """
        if not os.path.exists("../../results/train_test_splits/"):
            os.makedirs("../../results/train_test_splits/")

        self.mode = train_test
        self.cell_ids_test, self.cell_lines_test = (
            self.cell_ids_test_lists[fold],
            self.cell_lines_test_lists[fold],
        )

        self.cell_ids_train = np.setdiff1d(np.arange(self.molecular_data.shape[0]), self.cell_ids_test)
        self.cell_lines_train = self.molecular_data.index[self.cell_ids_train]

        self.cases_train = self.cases[np.isin(self.cases["cell_line"], self.cell_lines_train)]
        self.cases_test = self.cases[np.isin(self.cases["cell_line"], self.cell_lines_test)]

        self.cell_lines_train_filtered = np.array(self.cases_train["cell_line"].drop_duplicates())
        self.cell_lines_test_filtered = np.array(self.cases_test["cell_line"].drop_duplicates())

        self.current_cell_lines = (
            self.cell_lines_train_filtered if train_test == "train" else self.cell_lines_test_filtered
        )

        self.current_molecular_data = (
            self.molecular_data.loc[self.cell_lines_train_filtered, :]
            if train_test == "train"
            else self.molecular_data.loc[self.cell_lines_test_filtered, :]
        )

        if train_test == "train":
            self.scaler = StandardScaler().fit(self.current_molecular_data)

        self.current_molecular_data = pd.DataFrame(
            self.scaler.transform(self.current_molecular_data),
            index=self.current_molecular_data.index,
            columns=self.current_molecular_data.columns,
        )

        self.current_molecular_data_tensor = torch.tensor(np.array(self.current_molecular_data)).float()
        self.current_cases = (
            self.cases_train.reset_index(drop=True) if train_test == "train" else self.cases_test.reset_index(drop=True)
        )

        # save files
        test_names = pd.DataFrame({"cell_line": self.cell_lines_test_filtered})
        test_names["train_test"] = "test"

        train_names = pd.DataFrame({"cell_line": self.cell_lines_train_filtered})
        train_names["train_test"] = "train"

        names = pd.concat((test_names, train_names), axis=0)
        names["fold"] = fold

        names.to_csv("../../results/train_test_splits/train_test_names" + str(fold) + ".csv")

    def generate_train_and_test_ids(self) -> Tuple[List[torch.Tensor], List[List[Any]]]:
        """Generate train and test IDs for given cell names.

        Returns
        -------
            Tuple[List[torch.Tensor], List[List[Any]]]: A tuple containing the generated train and test IDs.
                The first element of the tuple is a list of PyTorch tensors containing train and test IDs,
                and the second element is a list of lists containing corresponding cell names for each ID split.
        """
        torch.manual_seed(0)
        all_ids = torch.randperm(self.molecular_data.index.shape[0])
        # Splits data into self.splits chunks
        id_split = np.array_split(all_ids, self.splits)
        logger.info(id_split)
        return id_split, [self.molecular_data.index[current] for current in id_split]

    def generate_train_val_test_ids(
        self, train_split: float = 0.7, val_split: float = 0.2, test_split: float = 0.1
    ) -> Tuple[List[torch.Tensor], List[List[Any]]]:
        """Generate train, validation, and test IDs for given cell-line names.

        Args:
            train_split (float): The percentage of data to allocate to the training set.
            val_split (float): The percentage of data to allocate to the validation set.
            test_split (float): The percentage of data to allocate to the test set.

        Returns
        -------
            Tuple[List[torch.Tensor], List[List[Any]]]: Tuple containing the generated train, validation, and test IDs.
                The first element of the tuple is a list of PyTorch tensors containing train, validation, and test IDs,
                and the second element is a list of lists containing corresponding cell names for each ID split.
        """
        # Ensure split percentages sum to 1.0
        assert abs(train_split + val_split + test_split - 1.0) < 1e-10, "Split percentages must sum to 1.0"

        # Generate random permutation of indices
        num_samples = len(self.molecular_data.index)
        all_ids = torch.randperm(num_samples)

        # Calculate number of samples for each split
        train_count = int(num_samples * train_split)
        val_count = int(num_samples * val_split)
        # test_count = num_samples - train_count - val_count

        # Split the indices into train, validation, and test sets
        train_ids = all_ids[:train_count]
        val_ids = all_ids[train_count : train_count + val_count]
        test_ids = all_ids[train_count + val_count :]

        # Create lists of cell names for each split
        train_names = [np.array(self.molecular_data.index)[i] for i in train_ids]
        val_names = [np.array(self.molecular_data.index)[i] for i in val_ids]
        test_names = [np.array(self.molecular_data.index)[i] for i in test_ids]

        # Return a tuple containing train, validation, and test IDs and names
        return [train_ids, val_ids, test_ids], [train_names, val_names, test_names]

    def get_drug_vector(self, current_drugname):
        """_summary_.

        Args:
            current_drugname (_type_): _description_

        Returns
        -------
            _type_: _description_
        """
        current_drug_embedding = self.drug_embeddings_tensor[
            :, current_drugname == self.drug_embeddings.columns
        ].squeeze()
        current_drug_fingerprint = (
            self.drug_fingerprints_tensor[:, self.drug_fingerprints.columns == current_drugname].squeeze()
            if current_drugname in self.drug_fingerprints.columns
            else torch.zeros(self.drug_fingerprints_tensor.shape[1]).squeeze()
        )

        current_drug_onehot = self.drug_onehot_tensor[:, self.unique_drugs_array == current_drugname].squeeze()

        return torch.cat(
            (current_drug_embedding, current_drug_fingerprint, current_drug_onehot),
            axis=0,
        )
