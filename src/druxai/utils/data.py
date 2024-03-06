"""Compound Data Module."""

import logging
import os

import torch as tc
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class MyDataSet(Dataset):
    """_summary_.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, file_path, n_splits, results_dir):
        self.file_path = file_path
        self.splits = n_splits
        self.results_dir = results_dir

        # Initialize logger
        self.logger = self._setup_logger()

        self.cases = pd.read_csv(
            os.path.join(file_path, "auc_secondary_screen_prediction_targets.csv"),
            index_col=0,
        ).reset_index(drop=True)
        self.molecular_data = pd.read_csv(os.path.join(file_path, "rna_df.csv"), index_col=0)
        self.unique_drugs, unique_cell_lines = (
            self.cases["DRUG"].drop_duplicates(),
            self.cases["cell_line"].drop_duplicates(),
        )
        self.unique_drugs_array = np.array(self.unique_drugs)
        self.drug_onehot_tensor = tc.eye(self.unique_drugs.shape[0])
        self.molecular_names = np.array(self.molecular_data.columns)

        os.makedirs(results_dir, exist_ok=True)
        pd.DataFrame({"names": self.molecular_names}).to_csv(os.path.join(results_dir, "correct_gene_order.csv"))

        self.cell_line = np.array(self.cases["cell_line"].drop_duplicates().sort_values())
        self.cell_ids_test_lists, self.cell_lines_test_lists = self.generate_train_and_test_ids(
            self.molecular_data.index
        )

        self.ndrug_features = self.unique_drugs.shape[0]
        self.nmolecular_features = self.molecular_data.shape[1]
        self.ncelltypes = self.cell_line.shape[0]

        self.logger.info(
            f"{self.nmolecular_features} molecular features, "
            f"{self.ncelltypes} celltypes, "
            f"{self.ndrug_features} drug_features"
        )
        self.logger.info(f"{self.unique_drugs.shape[0]} drugs and {unique_cell_lines.shape[0]} cell lines")

    def __getitem__(self, idx):
        """_summary_.

        Args:
            idx (_type_): _description_

        Returns
        -------
            _type_: _description_
        """
        case = self.current_cases.loc[idx, :]
        current_drugname = case["DRUG"]
        current_model_id = case["cell_line"]

        # only use this for now
        current_drug_onehot = self.drug_onehot_tensor[:, self.unique_drugs_array == current_drugname].squeeze()
        current_molecular_data = self.current_molecular_data_tensor[
            tc.tensor(self.current_molecular_data.index == current_model_id).squeeze(),
            :,
        ].squeeze()
        current_outcome = tc.tensor(case["auc_per_drug"]).float()

        return (
            current_drug_onehot,
            current_molecular_data,
            current_outcome.unsqueeze(0),
            idx,
        )

    def __len__(self):
        """_summary_.

        Returns
        -------
            _type_: _description_
        """
        return self.current_cases.shape[0]

    def _setup_logger(self):
        """_summary_.

        Returns
        -------
            _type_: _description_
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create file handler which logs messages to a file
        log_file = os.path.join("../../logs", "logfile.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter and add it to the file handler
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add file handler to the logger
        logger.addHandler(file_handler)

        return logger

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

        self.current_molecular_data_tensor = tc.tensor(np.array(self.current_molecular_data)).float()
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

    def generate_train_and_test_ids(self, cell_names):
        """_summary_.

        Args:
            cell_names (_type_): _description_

        Returns
        -------
            _type_: _description_
        """
        tc.manual_seed(0)
        all_ids = tc.randperm(cell_names.shape[0])
        id_split = np.array_split(all_ids, self.splits)
        return id_split, [cell_names[current] for current in id_split]

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
            else tc.zeros(self.drug_fingerprints_tensor.shape[1]).squeeze()
        )

        current_drug_onehot = self.drug_onehot_tensor[:, self.unique_drugs_array == current_drugname].squeeze()

        return tc.cat(
            (current_drug_embedding, current_drug_fingerprint, current_drug_onehot),
            axis=0,
        )
