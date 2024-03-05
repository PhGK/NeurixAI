import torch as tc
from druxai.utils.data import MyDataSet
from druxai.models.NN import Interaction_Model
import os
import pandas as pd
from druxai.utils.cross_validate import crossvalidate
import sys


print(sys.argv)
def main():
    if not os.path.exists('./results/data/'):
        os.makedirs('./results/data/')
    
    ds = MyDataSet(nsplits = 5)
    model_interaction = Interaction_Model(ds)

    # crossvalidate
    crossvalidate(model_interaction,ds, device = tc.device('cuda:0'))

if __name__ == '__main__':
    print('lets go')
    main()
    
