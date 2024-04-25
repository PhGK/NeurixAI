import torch as tc
from data import MyDataSet
#from training import train_test
from NN import Interaction_Model
import os
import pandas as pd
from crossant import crossvalidate
import sys


print(sys.argv)
def main():
    if not os.path.exists('./results/data/'):
        os.makedirs('./results/data/')

    
    ds = MyDataSet(nsplits = 5)

    ################
    #construct model
    ################

    model_interaction = Interaction_Model(ds)


    #################
    #crossvalidate
    #################
    crossvalidate(model_interaction,ds, device = tc.device('cuda:0'))



if __name__ == '__main__':
    print('lets go')
    main()
    
