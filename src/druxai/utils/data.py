"""Compound Data Module."""

import os
from typing import Any, Dict, List, Tuple

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
        self.selected_dataset = "train"
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

        self.train_test_ids_tensor_list, self.train_test_cell_names_list = self.generate_train_and_test_ids()

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
        if self.selected_dataset == "train":
            self.current_molecular_data_tensor = self.current_molecular_data_tensor_train
            self.current_molecular_data = self.current_molecular_data_train_pd
            self.current_cases = self.cases_train.reset_index(drop=True)
        elif self.selected_dataset == "test":
            self.current_molecular_data_tensor = self.current_molecular_data_tensor_test
            self.current_molecular_data = self.current_molecular_data_test_pd
            self.current_cases = self.cases_test.reset_index(drop=True)
        elif self.selected_dataset == "val":
            self.current_molecular_data_tensor = self.current_molecular_data_tensor_val
            self.current_molecular_data = self.current_molecular_data_val_pd
            self.current_cases = self.cases_val.reset_index(drop=True)
        else:
            raise ValueError("Invalid value for selected_dataset.")

        case = self.current_cases.loc[idx, :]
        drugnames = case["DRUG"]
        cell_line_names = case["cell_line"]

        current_drug_onehot = self.drug_onehot_tensor[:, self.unique_drugs_array == drugnames].squeeze()
        current_outcome = torch.tensor(case["auc_per_drug"]).float()

        # TODO: Check that return isnt empty when entire mask is false.
        mask = torch.tensor(self.current_molecular_data.index == cell_line_names).squeeze()
        current_molecular_data = self.current_molecular_data_tensor[mask, :].squeeze()

        return (current_drug_onehot, current_molecular_data, current_outcome.unsqueeze(0), idx)

    def __len__(self):
        """_summary_.

        Returns
        -------
            _type_: _description_
        """
        if self.selected_dataset == "train":
            return len(self.cases_train.reset_index(drop=True))
        if self.selected_dataset == "test":
            return len(self.cases_test.reset_index(drop=True))
        if self.selected_dataset == "val":
            return len(self.cases_val.reset_index(drop=True))
        raise ValueError("No length to return, since selected_dataset is not specified.")

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
            self.train_test_ids_tensor_list[fold],
            self.train_test_cell_names_list[fold],
        )

        # Find cell ids for train and test sets
        self.cell_ids_train = np.setdiff1d(np.arange(self.molecular_data.shape[0]), self.cell_ids_test)

        # Only use cell lines for which we have molecular data
        self.cell_lines_train = self.molecular_data.index[self.cell_ids_train]

        # Find for the cell line in training or test set all corresponding experiments (can be multiple per cell line)
        self.cases_train = self.cases[np.isin(self.cases["cell_line"], self.cell_lines_train)]
        self.cases_test = self.cases[np.isin(self.cases["cell_line"], self.cell_lines_test)]

        # TODO: Drop duplicates again (I think unnecessary since we handeld duplicates before already during import)
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
        return id_split, [self.molecular_data.index[current] for current in id_split]

    def generate_train_val_test_ids(
        self, train_split: float = 0.7, val_split: float = 0.2, test_split: float = 0.1, seed: int = None
    ) -> dict:
        """Generate train, validation, and test IDs for given cell-line names.

        Args:
            train_split (float): The percentage of data to allocate to the training set.
            val_split (float): The percentage of data to allocate to the validation set.
            test_split (float): The percentage of data to allocate to the test set.
            seed (int): The seed value for the random number generator.

        Returns
        -------
            dict: Dictionary containing the generated train, validation, and test IDs along with their names.
        """
        # Ensure split percentages sum to 1.0
        assert abs(train_split + val_split + test_split - 1.0) < 1e-10, "Split percentages must sum to 1.0"

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Generate random permutation of indices
        num_samples = len(self.molecular_data.index)
        all_ids = torch.randperm(num_samples)

        # Calculate number of samples for each split
        train_count = int(num_samples * train_split)
        val_count = int(num_samples * val_split)

        # Split the indices into train, validation, and test sets
        train_ids = all_ids[:train_count]
        val_ids = all_ids[train_count : train_count + val_count]
        test_ids = all_ids[train_count + val_count :]

        # Create lists of cell names for each split
        train_names = [np.array(self.molecular_data.index)[i] for i in train_ids]
        val_names = [np.array(self.molecular_data.index)[i] for i in val_ids]
        test_names = [np.array(self.molecular_data.index)[i] for i in test_ids]

        # Return a dictionary containing train, validation, and test IDs and names
        return {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "train_names": train_names,
            "val_names": val_names,
            "test_names": test_names,
        }

    def preprocess_train_val_test(self, ids: Dict[str, list]):
        """Preprocess training, validation, and test datasets.

        Extracts and Scales train, val and test data based on provided IDs.

        Args:
            ids (Dict[str, list]): Dictionary containing train, test, and val cell line names.

        Returns
        -------
            None
        """
        # Only use cell lines for which we have molecular data
        self.cell_lines_train = self.molecular_data.loc[ids["train_names"]]
        self.cell_lines_test = self.molecular_data.loc[ids["test_names"]]
        self.cell_lines_val = self.molecular_data.loc[ids["val_names"]]

        # Find for the cell line in training or test set all corresponding experiments (can be multiple per cell line)
        self.cases_train = self.cases[self.cases["cell_line"].isin(self.cell_lines_train.index)]
        self.cases_test = self.cases[self.cases["cell_line"].isin(self.cell_lines_test.index)]
        self.cases_val = self.cases[self.cases["cell_line"].isin(self.cell_lines_val.index)]

        # Filtered cell lines
        self.cell_lines_train_filtered = self.cases_train["cell_line"].unique()
        self.cell_lines_test_filtered = self.cases_test["cell_line"].unique()
        self.cell_lines_val_filtered = self.cases_val["cell_line"].unique()

        # Create scaler and fit it on training data
        self.scaler = StandardScaler().fit(self.molecular_data.loc[self.cell_lines_train_filtered])

        # Transform molecular data for train, test, and val sets using the fitted scaler; and save as pd Dataframe
        self.current_molecular_data_train_pd = pd.DataFrame(
            self.scaler.transform(self.molecular_data.loc[self.cell_lines_train_filtered]),
            index=self.cell_lines_train_filtered,
            columns=self.molecular_data.columns,
        )
        self.current_molecular_data_test_pd = pd.DataFrame(
            self.scaler.transform(self.molecular_data.loc[self.cell_lines_test_filtered]),
            index=self.cell_lines_test_filtered,
            columns=self.molecular_data.columns,
        )
        self.current_molecular_data_val_pd = pd.DataFrame(
            self.scaler.transform(self.molecular_data.loc[self.cell_lines_val_filtered]),
            index=self.cell_lines_val_filtered,
            columns=self.molecular_data.columns,
        )

        # Transform dataframes to torch tensor
        self.current_molecular_data_tensor_train = torch.tensor(np.array(self.current_molecular_data_train_pd)).float()
        self.current_molecular_data_tensor_test = torch.tensor(np.array(self.current_molecular_data_test_pd)).float()
        self.current_molecular_data_tensor_val = torch.tensor(np.array(self.current_molecular_data_val_pd)).float()

    def get_drug_vector(self, drugnames):
        """_summary_.

        Args:
            drugnames (_type_): _description_

        Returns
        -------
            _type_: _description_
        """
        current_drug_embedding = self.drug_embeddings_tensor[:, drugnames == self.drug_embeddings.columns].squeeze()
        current_drug_fingerprint = (
            self.drug_fingerprints_tensor[:, self.drug_fingerprints.columns == drugnames].squeeze()
            if drugnames in self.drug_fingerprints.columns
            else torch.zeros(self.drug_fingerprints_tensor.shape[1]).squeeze()
        )

        current_drug_onehot = self.drug_onehot_tensor[:, self.unique_drugs_array == drugnames].squeeze()

        return torch.cat(
            (current_drug_embedding, current_drug_fingerprint, current_drug_onehot),
            axis=0,
        )
