"""Training Model to train NN."""

import os

import torch as tc
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd


def train_test(model, ds, lr, nepochs, fold, device, PATH="../../results/training/"):
    """_summary_.

    Args:
        model (_type_): _description_
        ds (_type_): _description_
        lr (_type_): _description_
        nepochs (_type_): _description_
        fold (_type_): _description_
        device (_type_): _description_
        PATH (str, optional): _description_. Defaults to "../../results/training/".

    Returns
    -------
        _type_: _description_
    """
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    model.train().to(device)

    optimizer1 = SGD(model.nn1.parameters(), momentum=0.9, lr=lr, weight_decay=1e-5)
    optimizer2 = SGD(model.nn2.parameters(), momentum=0.9, lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss()

    scheduler1 = ExponentialLR(optimizer1, gamma=0.9)  # gamma 0.9 #0.95 only for 100 epochs
    scheduler2 = ExponentialLR(optimizer2, gamma=0.9)  # gamma 0.9

    ds.change_fold(fold, "train")
    for epoch in range(1, nepochs + 1):  # been without +1
        bs = 128
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)

        model.train()
        for drug, molecular, outcome, _, it in enumerate(tqdm(dl)):
            assert ds.mode == "train", "training on " + ds.mode + " dataset is not possible"
            if it > 50:
                pass

            drug, molecular, outcome = (
                drug.to(device),
                molecular.to(device),
                outcome.to(device),
            )

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            prediction = model.forward(drug, molecular)

            loss = criterion(outcome, prediction)

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)

            if tc.rand(1) < 0.5:
                optimizer1.step()
            else:
                optimizer2.step()

        if epoch in range(0, 20, 1):
            ds.change_fold(fold, "test")
            assert ds.mode == "test", "testing on " + ds.mode + " dataset is not possible"

            dl_test = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)

            testlosses = []
            prediction_frames = []
            model.eval()
            for drug, molecular, outcome, idx in tqdm(dl_test):
                drug, molecular, outcome = (
                    drug.to(device),
                    molecular.to(device),
                    outcome.to(device),
                )

                with tc.no_grad():
                    prediction = model.forward(drug, molecular)

                    testlosses.append(criterion(outcome, prediction))

                    batch_result = pd.DataFrame(
                        {
                            "ground_truth": outcome.squeeze().cpu().numpy(),
                            "prediction": prediction.squeeze().cpu().numpy(),
                        }
                    )

                    batch_result["cells"] = np.array(ds.current_cases.loc[np.array(idx), "cell_line"])
                    batch_result["drugs"] = np.array(ds.current_cases.loc[np.array(idx), "DRUG"])
                    batch_result["fold"] = fold
                    batch_result["epoch"] = epoch

                    prediction_frames.append(batch_result)
                # break

            if epoch % 1 == 0:
                pd.concat(prediction_frames, axis=0).to_csv(
                    os.path.join(
                        PATH,
                        "prediction_frame_epoch" + str(epoch) + "_fold" + str(fold) + ".csv",
                    )
                )

            ds.change_fold(fold, "train")

        scheduler1.step()
        scheduler2.step()

    return model
