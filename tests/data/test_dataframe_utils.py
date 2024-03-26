"""Tests for python dataset class."""

import pytest

import pandas as pd

from druxai.utils.dataframe_utils import split_data_by_cell_line_ids


def test_disjoint_splits():
    """
    Test function to check for disjoint splits based on cell line IDs.

    This function creates a sample DataFrame with cell line IDs and then splits the data using
    the split_data_by_cell_line_ids function.
    It checks if there are any overlapping cell line IDs between splits and asserts that there are none.

    Raises
    ------
        AssertionError: If there are overlapping cell line IDs between splits.

    Returns
    -------
        None
    """
    data = {"cell_line": ["1", "1", "2", "2", "3", "3", "4", "4"], "feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
    df = pd.DataFrame(data)

    # Perform the split
    train_ids, val_ids, test_ids = split_data_by_cell_line_ids(df)

    # Combine the IDs for all splits
    all_ids = train_ids + val_ids + test_ids
    # Check for duplicates
    duplicates = [id for id in all_ids if all_ids.count(id) > 1]
    # Assert no duplicates
    assert not duplicates, "There are overlapping cell line IDs between splits."


# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
