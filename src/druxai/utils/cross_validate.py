from models.LRP import calculate_one_side_LRP
import torch as tc
from models.NN import Interaction_Model
import os
from utils.data import LRPSet


def crossvalidate(model, ds, device, results_dir="../../results"):
    """_summary_

    Args:
        model (_type_): _description_
        ds (_type_): _description_
        device (_type_): _description_
        results_dir (str, optional): _description_. Defaults to "../../results".
    """

    os.makedirs(os.path.join(results_dir, 'model_params'), exist_ok=True)
    tc.save(model.state_dict(), './results/raw_params.pt')

    for i in [4]:  # range(ds.splits):
        model = Interaction_Model(ds)
        # model.load_state_dict(tc.load('./results/raw_params.pt'))
        if True:
            # Calculation of all LRP values
            LRP_ds = LRPSet(n_splits=5)
            LRP_ds.change_fold(i, 'train')  # just to calculate scaling parameters
            LRP_ds.change_fold(i, 'test')
            model.load_state_dict(
                tc.load(
                    './results/model_params/model_params_fold' +
                    str(i) +
                    '.pt'))
            calculate_one_side_LRP(model, LRP_ds, fold=i)
