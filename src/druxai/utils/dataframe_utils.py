"""Module contains functions for creating DataFrame from computed variables."""

from typing import Any, List, Tuple

from torch import Tensor

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from druxai.utils.data import DrugResponseDataset


def create_batch_result(outcome: Tensor, prediction: Tensor, ds: Any, idx: Any, epoch: int) -> DataFrame:
    """
    Create a DataFrame containing batch results.

    Args:
        outcome (torch.Tensor): Ground truth outcomes.
        prediction (torch.Tensor): Predicted outcomes.
        ds (Any): dataset
        idx (Any): Description of idx parameter.
        fold (int): Fold number.
        epoch (int): Epoch number.

    Returns
    -------
        pd.DataFrame: DataFrame containing batch results.
    """
    return pd.DataFrame(
        {
            "ground_truth": outcome.squeeze().cpu().numpy(),
            "prediction": prediction.squeeze().cpu().numpy(),
            "cells": np.array(ds.targets.iloc[idx]["cell_line"]),
            "drugs": np.array(ds.targets.iloc[idx]["DRUG"]),
            "epoch": epoch,
        }
    )


# TODO: Write test to check that splits are actually unique for the cell lines!
def split_data_by_cell_line_ids(
    dataframe: pd.DataFrame, train_pct: float = 0.7, test_pct: float = 0.15, val_pct: float = 0.15, seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split the DataFrame into train, test, and validation sets based.

    It splits on cell line IDs ensuring that each cell line is only assigned to one set.

    Args
    ----------
    dataframe (pd.DataFrame): Input DataFrame containing data to split.
    train_pct (float): Percentage of data to allocate to the train set (default is 0.7).
    test_pct (float): Percentage of data to allocate to the test set (default is 0.15).
    val_pct (float): Percentage of data to allocate to the validation set (default is 0.15).
    seed (int or None): Seed for random number generation (default is 42).

    Returns
    -------
    Tuple[List[int], List[int], List[int]]: Tuple containing lists of IDs for the train, validation, and test sets.
    """
    # Set the seed for random number generation
    rng = np.random.default_rng(seed)

    # Step 1: Group by cell line
    grouped = dataframe.groupby("cell_line")

    # Step 2: Initialize sets to keep track of assigned cell line IDs
    assigned_train, assigned_test, assigned_validation = set(), set(), set()

    # Step 3: Split each group into train, test, and validation sets
    train_dfs, test_dfs, validation_dfs = [], [], []
    for _, group_df in grouped:
        # Check if the cell line ID has been assigned before
        cell_line_id = group_df["cell_line"].iloc[0]  # Assuming 'cell_line' is the first column
        if (
            cell_line_id not in assigned_train
            and cell_line_id not in assigned_test
            and cell_line_id not in assigned_validation
        ):
            # Assign the cell line ID to a set based on a random draw
            draw = rng.random()
            if draw < train_pct:  # train_pct chance for train set
                assigned_train.add(cell_line_id)
                train_dfs.append(group_df)
            elif draw < train_pct + test_pct:  # test_pct chance for test set
                assigned_test.add(cell_line_id)
                test_dfs.append(group_df)
            else:  # val_pct chance for validation set
                assigned_validation.add(cell_line_id)
                validation_dfs.append(group_df)

    # Step 4: Concatenate the sets to create the final train, test, and validation DataFrames
    train_ids = pd.concat(train_dfs).index.to_list()
    val_ids = pd.concat(validation_dfs).index.to_list()
    test_ids = pd.concat(test_dfs).index.to_list()

    return train_ids, val_ids, test_ids


def standardize_molecular_data_inplace(
    data: DrugResponseDataset, train_id: List, val_id: List, test_id: List, scaler_type: str = "StandardScaler"
) -> None:
    """
    Standardizes molecular data in the dataset object in place based on provided train, validation, and test IDs.

    Args:
        data (DrugResponseDataset): Dataset object containing molecular data.
        train_id (Any): IDs for training set.
        val_id (Any): IDs for validation set.
        test_id (Any): IDs for testing set.
        scaler_type (str, optional): Type of scaler to use. Options: 'StandardScaler', 'MinMaxScaler', 'RobustScaler'.
                                        Defaults to 'StandardScaler'.
    """
    # Select scaler based on scaler_type
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaler_type. Choose from 'StandardScaler', 'MinMaxScaler', or 'RobustScaler'.")

    # Standartize gene data
    scaler = StandardScaler()
    train_molecular_data = data.molecular_data.loc[data.targets.iloc[train_id]["cell_line"].unique()]
    scaler.fit(train_molecular_data.values)
    standardized_train_molecular_data = scaler.transform(train_molecular_data.values)

    # Apply the scaler to the validation and testing sets
    val_molecular_data = data.molecular_data.loc[data.targets.iloc[val_id]["cell_line"].unique()]
    standardized_val_molecular_data = scaler.transform(val_molecular_data.values)

    test_molecular_data = data.molecular_data.loc[data.targets.iloc[test_id]["cell_line"].unique()]
    standardized_test_molecular_data = scaler.transform(test_molecular_data.values)

    # Convert standardized arrays back to DataFrame
    standardized_train_molecular_df = pd.DataFrame(
        standardized_train_molecular_data, index=train_molecular_data.index, columns=train_molecular_data.columns
    )
    standardized_val_molecular_df = pd.DataFrame(
        standardized_val_molecular_data, index=val_molecular_data.index, columns=val_molecular_data.columns
    )
    standardized_test_molecular_df = pd.DataFrame(
        standardized_test_molecular_data, index=test_molecular_data.index, columns=test_molecular_data.columns
    )

    # Update molecular data in the dataset object with standardized DataFrames
    data.molecular_data.loc[data.targets.iloc[train_id]["cell_line"].unique()] = standardized_train_molecular_df
    data.molecular_data.loc[data.targets.iloc[val_id]["cell_line"].unique()] = standardized_val_molecular_df
    data.molecular_data.loc[data.targets.iloc[test_id]["cell_line"].unique()] = standardized_test_molecular_df
