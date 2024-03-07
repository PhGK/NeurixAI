"""Module contains functions for creating DataFrame from computed variables."""

from typing import Any

from torch import Tensor

import numpy as np
import pandas as pd
from pandas import DataFrame


def create_batch_result(outcome: Tensor, prediction: Tensor, ds: Any, idx: Any, fold: int, epoch: int) -> DataFrame:
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
            "cells": np.array(ds.current_cases.loc[np.array(idx), "cell_line"]),
            "drugs": np.array(ds.current_cases.loc[np.array(idx), "DRUG"]),
            "fold": fold,
            "epoch": epoch,
        }
    )
