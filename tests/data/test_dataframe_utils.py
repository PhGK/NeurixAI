"""Tests for python dataset class."""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import pandas as pd

from druxai.utils.data import DataloaderSampler
from druxai.utils.dataframe_utils import split_data_by_cell_line_ids


class MockDrugResponseDataset(Dataset):
    """Test Drug Response Dataset class for testing the DrugResponseDataset class."""

    def __init__(self, num_samples=10):
        cell_lines = [f"cell_{i}" for i in range(num_samples)]
        drugs = [f"drug_{i}" for i in range(num_samples)]
        targets = torch.randn(num_samples)

        # Create pairings of each cell line with each drug
        cell_lines = [cell for cell in cell_lines for _ in range(num_samples)]  # repeat each cell line
        drugs = drugs * num_samples  # repeat drugs to match cell line repetitions
        targets = torch.randn(num_samples * num_samples)  # targets for each pairing

        self.targets = pd.DataFrame({"cell_line": cell_lines, "DRUG": drugs, "target": targets})

        self.molecular_data = torch.randn(num_samples * num_samples, 10)  # 10 molecular features

        self.drug_encoding_dict = {
            f"drug_{i}": torch.zeros(num_samples * num_samples).scatter_(0, torch.tensor(i), 1)
            for i in range(num_samples)
        }

    def __getitem__(self, idx):
        """Mock getitem magic function like in DrugResponseDataset."""
        gene_expression_values = self.molecular_data[idx]
        target = torch.tensor([self.targets.iloc[idx]["target"]], dtype=torch.float32)
        drug_one_hot = self.drug_encoding_dict[self.targets.iloc[idx]["DRUG"]]
        return {"gene_expression": gene_expression_values, "drug_encoding": drug_one_hot}, target, idx

    def __len__(self):
        """Return the length of the targets."""
        return len(self.targets)


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
    df = MockDrugResponseDataset()

    # Perform the split
    train_ids, val_ids, test_ids = split_data_by_cell_line_ids(df.targets)

    # Combine the IDs for all splits
    all_ids = train_ids + val_ids + test_ids
    # Check for duplicates
    duplicates = [id for id in all_ids if all_ids.count(id) > 1]
    # Assert no duplicates
    assert not duplicates, "There are overlapping cell line IDs between splits."


@pytest.mark.parametrize("batch_size", [1, 5, 10])
@pytest.mark.parametrize("num_samples", [10, 50, 100])
def test_epoch_batches(batch_size, num_samples):
    """
    Test whether the batch splits repeat after each epoch in the DataLoader.

    This function creates a sample dataset and performs a split using a custom sampler.
    It then creates a DataLoader with the custom sampler and tracks the batches seen in each epoch.
    After each epoch, it checks if the batch splits repeat, ensuring that the same elements appear
    in the same batch after each epoch.
    """
    # Instantiate the sample dataset
    df = MockDrugResponseDataset(num_samples=num_samples)

    # Perform the split
    fixed_indices, _, _ = split_data_by_cell_line_ids(df.targets)

    # Create DataLoader with custom sampler
    custom_sampler = DataloaderSampler(fixed_indices)
    dataloader = DataLoader(df, batch_size=batch_size, sampler=custom_sampler)

    # Track batches seen in each epoch
    epoch_batches = []

    # Iterate over epochs
    for epoch in range(3):  # assuming 3 epochs for testing
        batches_seen = []
        for _, _, sample_id in dataloader:
            batches_seen.append(sample_id.tolist())

        # Check if batch splits repeat after each epoch
        if epoch > 0:
            # Compare sample IDs between current and previous epoch
            assert batches_seen != epoch_batches

        # Save batches seen in this epoch
        epoch_batches = batches_seen


# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
