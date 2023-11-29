import torch as tc
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import os


class MyDataSet(Dataset):
    def __init__(self, nsplits):
        self.splits = nsplits
        
        self.cases = pd.read_csv('./use_data/prediction_targets.csv', index_col=0).reset_index(drop=True)
        self.unique_drugs, unique_cell_lines = self.cases['DRUG'].drop_duplicates(),self.cases['cell_line'].drop_duplicates()
        self.unique_drugs_array = np.array(self.unique_drugs)


        self.drug_embeddings = pd.read_csv('./use_data/embeddings.csv', index_col=0)
        self.drug_fingerprints = pd.read_csv('./use_data/ECFP.csv', index_col=0).transpose()

        self.drug_embeddings_tensor = tc.tensor(np.array(self.drug_embeddings)).float()        
        self.drug_fingerprints_tensor = tc.tensor(np.array(self.drug_fingerprints)).float()
        self.drug_onehot_tensor = tc.eye(self.unique_drugs.shape[0])


        self.molecular_data = pd.read_csv('./use_data/rna_data.csv', index_col=0)
        self.molecular_names = np.array(self.molecular_data.columns)
        pd.DataFrame({'names': self.molecular_names}).to_csv('./results/correct_gene_order.csv')


        self.cell_line = np.array(self.cases['cell_line'].drop_duplicates().sort_values())

        self.cell_ids_test_lists, self.cell_lines_test_lists = self.generate_train_and_test_ids(self.molecular_data.index) 
        

        #self.ndrug_features = self.drug_embeddings.shape[0] + self.drug_fingerprints.shape[0]   + self.unique_drugs.shape[0]
        self.ndrug_features = self.drug_fingerprints.shape[0]   + self.unique_drugs.shape[0]
        self.nmolecular_features  = self.molecular_data.shape[1]
        self.ncelltypes = self.cell_line.shape[0]
        

        print(str(self.nmolecular_features) + ' molecular features, ' + str(self.ncelltypes) + ' celltypes, ' + str(self.ndrug_features) + ' drug_features')
        print(str(self.unique_drugs.shape[0]) + 'drugs and ' + str(unique_cell_lines.shape[0]) + 'cell lines')



    def change_fold(self,fold, train_test):
        if not os.path.exists('./results/train_test_splits/'):
            os.makedirs('./results/train_test_splits/')

        self.mode = train_test
        self.cell_ids_test, self.cell_lines_test = self.cell_ids_test_lists[fold], self.cell_lines_test_lists[fold]
        
        self.cell_ids_train  = np.setdiff1d(np.arange(self.molecular_data.shape[0]), self.cell_ids_test)
        self.cell_lines_train= self.molecular_data.index[self.cell_ids_train]
        
        self.cases_train = self.cases[np.isin(self.cases['cell_line'], self.cell_lines_train)]
        self.cases_test = self.cases[np.isin(self.cases['cell_line'], self.cell_lines_test)]
        
        print(self.cases_train.shape, self.cases_test.shape)
        
        self.current_cell_lines = self.cell_lines_train if train_test == 'train' else self.cell_lines_test
        self.current_molecular_data = self.molecular_data.iloc[np.array(self.cell_ids_train),:] if train_test == 'train' else self.molecular_data.iloc[np.array(self.cell_ids_test),:]
        
        if train_test == 'train':
            self.scaler = StandardScaler().fit(self.current_molecular_data)
            
        self.current_molecular_data = pd.DataFrame(self.scaler.transform(self.current_molecular_data), index = self.current_molecular_data.index, columns = self.current_molecular_data.columns)
        
        self.current_molecular_data_tensor = tc.tensor(np.array(self.current_molecular_data)).float()
        self.current_cases = self.cases_train.reset_index(drop=True) if train_test == 'train' else self.cases_test.reset_index(drop=True)
        
        #save files
        test_names = pd.DataFrame({'cell_line': self.cell_lines_test})
        test_names['train_test'] = 'test'
        
        train_names = pd.DataFrame({'cell_line': self.cell_lines_train})
        train_names['train_test'] = 'train'
        
        names = pd.concat((test_names, train_names), axis=0)
        names['fold'] = fold
        
        names.to_csv('./results/train_test_splits/train_test_names' + str(fold)+ '.csv')
        
        

    def __len__(self):
        return self.current_cases.shape[0]


    def __getitem__(self,idx):
        
        case = self.current_cases.loc[idx,:]

        
        current_drugname = case['DRUG']
        current_model_id = case['cell_line']
        
        
        ##node2vec drug embedding
        #current_drug_embedding = self.drug_embeddings_tensor[:,current_drugname== self.drug_embeddings.columns].squeeze()
        current_drug_fingerprint = self.drug_fingerprints_tensor[:,self.drug_fingerprints.columns == current_drugname].squeeze() if current_drugname in self.drug_fingerprints.columns else tc.zeros(self.drug_fingerprints_tensor.shape[1]).squeeze()
        

        current_drug_onehot = self.drug_onehot_tensor[:,self.unique_drugs_array == current_drugname].squeeze()
        

        #current_drug_embedding_with_dose = tc.cat((current_drug_embedding, current_drug_fingerprint,current_drug_onehot), axis=0)
        current_drug_embedding_with_dose = tc.cat((current_drug_fingerprint,current_drug_onehot), axis=0)

        current_molecular_data = self.current_molecular_data_tensor[tc.tensor(self.current_molecular_data.index == current_model_id).squeeze(),:].squeeze()

        current_outcome = tc.tensor(case['auc_per_drug']).float() 

        return current_drug_embedding_with_dose, current_molecular_data, current_outcome.unsqueeze(0), idx



    def generate_train_and_test_ids(self, cell_names):         
        tc.manual_seed(0)
        all_ids = tc.randperm(cell_names.shape[0])
        id_split = np.array_split(all_ids, self.splits)
        return id_split, [cell_names[current] for current in id_split]


class LRPSet(MyDataSet):
    def change_fold(self,fold, train_test):
        super().change_fold(fold, train_test)
        self.chosen_drugs = ['POZIOTINIB', 'TRAMETINIB', 'COBIMETINIB', 'DACOMITINIB', 'PELITINIB', 'IBRUTINIB', 'IDASANUTLIN', 'VINCRISTINE', 'SELUMETINIB','CANERTINIB', 'OSIMERTINIB' 'UPROSERTIB', 'DASATINIB', 'LAPATINIB', 'VINBLASTINE', 'OSIMERTINIB']
        self.current_cases = self.current_cases[np.isin(self.current_cases['DRUG'], self.chosen_drugs)].reset_index(drop=True)
        
        print(self.current_cases)
        print('a')
        self.current_cases = self.current_cases[['DRUG', 'cell_line', 'auc_per_drug']].groupby(['DRUG', 'cell_line'])['auc_per_drug'].mean().reset_index()
        print(self.current_cases)

        

