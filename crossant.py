from training import train_test
from LRP import calculate_one_side_LRP
import pandas as pd
import numpy as np
import copy
import torch as tc
from NN import Interaction_Model
import os
from data import LRPSet


def crossvalidate(model,ds, device):
    if not os.path.exists('./results/model_params/'):
        os.makedirs('./results/model_params/')

    tc.save(model.state_dict(), './results/raw_params.pt')

    for i in range(ds.splits):


        model = Interaction_Model(ds)
        #model.load_state_dict(tc.load('./results/raw_params.pt'))


        if True:
            model = train_test(model, ds, lr=3e-4,nepochs=51, fold=i, device=device)
            tc.save(model.state_dict(), './results/model_params/model_params_fold'+ str(i) +'.pt')
            
        
        if False:
            LRP_ds = LRPSet(nsplits = 5)
            LRP_ds.change_fold(i, 'test')
            model.load_state_dict(tc.load('./results/model_params/model_params_fold'+ str(i) +'.pt'))
            calculate_one_side_LRP(model, LRP_ds, fold=i)

            





