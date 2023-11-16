from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
import torch.nn as nn
import torch as tc
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from NN import Interaction_Model
import gc
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from sklearn.preprocessing import RobustScaler

        
def train_test(model, ds, lr, nepochs, fold, device, PATH = './results/training/'):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        
    print('fold: ',fold)

    model.train().to(device)

    optimizer1 = SGD(model.nn1.parameters(), momentum=0.9, lr = 0.05, weight_decay=1e-5)
    optimizer2 = SGD(model.nn2.parameters(), momentum=0.9, lr = 0.05, weight_decay=1e-5)
    criterion = nn.HuberLoss() #nn.L1Loss() #MSELoss()

    
    scheduler1 = ExponentialLR(optimizer1, gamma=0.9) #gamma 0.8
    scheduler2 = ExponentialLR(optimizer2, gamma=0.9) #gamma 0.8


    ds.change_fold(fold, 'train')
    for epoch in (range(nepochs)):

        

        
        bs = 128 #min(int(16*1.5**epoch), 512)
        
        dl = DataLoader(ds, batch_size = bs, shuffle=True, num_workers=8, pin_memory=True) #batch size 64


        it = 0
        model.train()
        for drug, molecular, outcome, idx in tqdm(dl):
           assert ds.mode == 'train', 'training on ' + ds.mode + ' dataset is not possible'
           it +=1
           if it>50:
               pass

        
           drug, molecular, outcome = drug.to(device), molecular.to(device), outcome.to(device)
           
           optimizer1.zero_grad()
           optimizer2.zero_grad()

           prediction = model.forward(drug, molecular)
           
           loss = criterion(outcome, prediction) 

           loss.backward()
           clip_grad_norm_(model.parameters(), 1.0)

           if tc.rand(1)<0.5:
               optimizer1.step()
           else:
               optimizer2.step()           

           
        if epoch in [5,10,30,50,80,100]:#epoch%5==0:
            print('batch size: ',bs, optimizer1.param_groups[0]['lr'])

            ds.change_fold(fold, 'test')
            assert ds.mode == 'test', 'testing on ' + ds.mode + ' dataset is not possible'

            dl_test = DataLoader(ds, batch_size = 1000, shuffle=False, num_workers=8,pin_memory=True)

            testlosses = []
            prediction_frames = []
            model.eval()
            for drug, molecular, outcome, idx in tqdm(dl_test):

                drug, molecular, outcome = drug.to(device), molecular.to(device), outcome.to(device)

                with tc.no_grad():
                    prediction = model.forward(drug, molecular)

                    testlosses.append(criterion(outcome, prediction))

                    batch_result = pd.DataFrame({'ground_truth': outcome.squeeze().cpu().numpy(), 'prediction': prediction.squeeze().cpu().numpy()})

                    batch_result['cells'] = np.array(ds.current_cases.loc[np.array(idx), 'cell_line'])
                    batch_result['drugs'] = np.array(ds.current_cases.loc[np.array(idx), 'DRUG'])
                    batch_result['fold'] = fold
                    batch_result['epoch'] = epoch

                    prediction_frames.append(batch_result)
                #break
                    
     
            if epoch%1==0:
                pd.concat(prediction_frames,axis=0).to_csv('./results/training/' + '/prediction_frame_epoch' + str(epoch) +'fold'+ str(fold) + '.csv')
                
            print(epoch, 'testloss:', tc.tensor(testlosses).mean())
            ds.change_fold(fold, 'train')
            
           
        scheduler1.step()
        scheduler2.step()

    return model





