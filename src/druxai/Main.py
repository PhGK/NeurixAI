"""Module to run model directly in terminal."""

import os

import torch as tc

from druxai.models.NN import Interaction_Model
from druxai.utils.cross_validate import crossvalidate
from druxai.utils.data import MyDataSet


def main():
    """_summary_."""
    if not os.path.exists("./results/data/"):
        os.makedirs("./results/data/")

    ds = MyDataSet(nsplits=5)
    model_interaction = Interaction_Model(ds)

    # crossvalidate
    crossvalidate(model_interaction, ds, device=tc.device("cuda:0"))


if __name__ == "__main__":
    """_summary_."""
    main()
