from training import train_test
from LRP import calculate_one_side_LRP, calculate_one_side_LRP_specific_genes
import pandas as pd
import numpy as np
import copy
import torch as tc
from NN import Interaction_Model
import os
from data import LRPSet,MyDataSet, LRPSet_CellLines


def crossvalidate(model,ds, device):
    if not os.path.exists('./results/model_params/'):
        os.makedirs('./results/model_params/')

    tc.save(model.state_dict(), './results/raw_params.pt')

    for i in [4]: #range(ds.splits):


        model = Interaction_Model(ds)
        #model.load_state_dict(tc.load('./results/raw_params.pt'))


        if False:
            model = train_test(model, ds, lr=0.05,nepochs=50, fold=i, device=device) #been 51 epochs, then 50
            tc.save(model.state_dict(), './results/model_params/model_params_fold'+ str(i) +'.pt')
            
        
        if True:
            LRP_ds = LRPSet(nsplits = 5)
            LRP_ds.change_fold(i, 'train') #just to calculate scaling parameters
            LRP_ds.change_fold(i, 'test')
            model.load_state_dict(tc.load('./results/model_params/model_params_fold'+ str(i) +'.pt'))
            calculate_one_side_LRP(model, LRP_ds, fold=i)
            
        if False:
            LRP_ds = LRPSet_CellLines(nsplits = 5)
            LRP_ds.change_fold(i, 'train') #just to calculate scaling parameters
            LRP_ds.change_fold(i, 'test')
            model.load_state_dict(tc.load('./results/model_params/model_params_fold'+ str(i) +'.pt'))
            calculate_one_side_LRP(model, LRP_ds, fold=i, PATH = './results/LRP_chosen_cell_lines/')
            
            
            
        if False:
            #specific_genes = np.array(pd.read_csv('./results/important_genes.csv')['molecular_names'])
            specific_genes = np.array(['ABCB1'])
            print(specific_genes)
            
            ds = MyDataSet(nsplits = 5)
            
            ds.change_fold(i, 'train') #just to calculate scaling parameters
            ds.change_fold(i, 'test')
            model.load_state_dict(tc.load('./results/model_params/model_params_fold'+ str(i) +'.pt'))
            calculate_one_side_LRP_specific_genes(model, ds,specific_genes = specific_genes, fold=i)

            




            





