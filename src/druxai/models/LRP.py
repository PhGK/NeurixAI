from torch.utils.data import DataLoader
import torch as tc
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
# from NN import Interaction_Model
from models.NN import LRP_Linear
import torch.nn as nn


def reverse_feature_expansion_interaction(frame):

    last = '_rev'

    LRP_frame = frame[['LRP', 'therapy', 'diagnostics', 'sample_name']]
    LRP_frame['therapy'] = LRP_frame['therapy'].str.split('_rev').str[0].copy()
    LRP_frame['diagnostics'] = LRP_frame['diagnostics'].str.split('_rev').str[0].copy()
    frame_unexpanded = LRP_frame.groupby(['therapy', 'diagnostics', 'sample_name'])[
        'LRP'].sum().reset_index()

    return frame_unexpanded


def np2pd(nparray, current_cell, current_drug, molecular_names):
    tensor = nparray
    ntensor_features = tensor.shape[1]

    unexpanded_tensor = tensor.copy()

    frame = pd.DataFrame(unexpanded_tensor, columns=molecular_names)

    frame['DRUG'] = np.array(current_drug)
    frame['cell_line'] = np.array(current_cell)
    long_frame = pd.melt(
        frame,
        id_vars=[
            'DRUG',
            'cell_line'],
        var_name='molecular_names',
        value_name='LRP')

    return long_frame


def generate_inputs(therapy_of_one_sample, diagnostics_of_one_sample):
    assert therapy_of_one_sample.name == diagnostics_of_one_sample.name, 'something is off'

    th = pd.DataFrame({'therapy_input': therapy_of_one_sample,
                      'therapy': therapy_of_one_sample.index})

    dia = pd.DataFrame({'diagnostics_input': diagnostics_of_one_sample,
                       'diagnostics': diagnostics_of_one_sample.index})

    merged = th.merge(dia, how='cross')
    merged['sample_name'] = therapy_of_one_sample.name
    return merged


def calculate_LRP_interaction(model, ds, output_feature=0, PATH='./results/LRP/', fold=None):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    device = tc.device('cuda:0')
    ds.generate_train_or_test_cases('test')
    model.eval().to(device)
    assert ds.mode == 'test', 'LRP on ' + ds.mode + ' dataset is not possible'
    dl = DataLoader(ds, batch_size=2000, shuffle=False)

    all_test_results = []
    it = 0
    for drug, molecular, outcome, idx in tqdm(dl):
        drug, molecular, outcome = drug.to(device), molecular.to(device), outcome.to(device)

        pred = model.get_product(drug, molecular)

        input_relevance = model.relprop(pred).cpu().detach().numpy()

        LRP_scores_long = np2pd(input_relevance,
                                current_cell=ds.cases_now.iloc[idx.cpu().numpy(),
                                                               :]['cell_line'],
                                current_drug=ds.cases_now.iloc[idx.cpu().numpy(),
                                                               :]['DRUG'],
                                molecular_names=ds.molecular_names)

        LRP_scores_long = LRP_scores_long.reset_index(drop=True)

        molecular_input = pd.DataFrame(molecular.cpu().numpy(), columns=ds.molecular_names)
        molecular_input['cell_line'] = np.array(
            ds.cases_now.iloc[idx.cpu().numpy(), :]['cell_line'])
        molecular_input['DRUG'] = np.array(ds.cases_now.iloc[idx.cpu().numpy(), :]['DRUG'])

        molecular_input_long = pd.melt(
            molecular_input,
            id_vars=[
                'DRUG',
                'cell_line'],
            var_name='molecular_names',
            value_name='expression')

        result_long = LRP_scores_long.merge(molecular_input_long, how='left')

        result_long.to_csv(
            PATH + 'LRP_' + str(fold) + '.csv',
            mode='w' if it == 0 else 'a',
            header=it == 0)
        it += 1

    # LRP_scores_and_inputs_all = pd.concat(all_test_results, axis=0)
    # LRP_scores_and_inputs_all.to_feather(PATH + 'LRP'+ str(model.classname) + '_interaction' + str(fold) + '.ftr')
    # LRP_scores_and_inputs_all.to_csv(PATH + 'LRP_' + str(fold) + '.csv')


def calculate_one_side_LRP(model, ds, output_feature=0, PATH='./results/LRP/', fold=None):

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    device = tc.device('cuda:0')
    # ds.generate_train_or_test_cases('test')
    model.eval().to(device)

    assert ds.mode == 'test', 'LRP on ' + ds.mode + ' dataset is not possible'

    dl = DataLoader(ds, batch_size=10, shuffle=False, num_workers=8, pin_memory=True)

    it = 0
    for drug, molecular, outcome, idx in tqdm(dl):
        drug, molecular, outcome = drug.to(device), molecular.to(device), outcome.to(device)

        drug_latent = model.nn1.forward(drug)
        molecular_latent = model.nn2.forward(molecular)
        # print(model.nn2.layers[1].A_dict)

        s = drug_latent.shape
        helper_network = LRP_Linear(s[0] * s[1], s[0] * s[1]).to(device).eval()

        helper_network.linear.weight = nn.Parameter(drug_latent.detach().view(
            s[0] * s[1]) * tc.eye(s[0] * s[1], device=device), requires_grad=True)
        helper_network.linear.bias = nn.Parameter(
            tc.zeros_like(helper_network.linear.bias), requires_grad=True)

        pred = helper_network.forward(molecular_latent.view(s[0] * s[1]))  # .view(s[0],s[1])

        latent_relevance = helper_network.relprop(pred).view(s[0], s[1])

        input_relevance = model.nn2.relprop(latent_relevance).cpu().detach().numpy()

        LRP_scores_long = np2pd(input_relevance,
                                current_cell=ds.current_cases.iloc[idx.cpu().numpy(),
                                                                   :]['cell_line'],
                                current_drug=ds.current_cases.iloc[idx.cpu().numpy(),
                                                                   :]['DRUG'],
                                molecular_names=ds.molecular_names)

        LRP_scores_long = LRP_scores_long.reset_index(drop=True)

        molecular_input = pd.DataFrame(molecular.cpu().numpy(), columns=ds.molecular_names)
        molecular_input['cell_line'] = np.array(
            ds.current_cases.iloc[idx.cpu().numpy(), :]['cell_line'])
        molecular_input['DRUG'] = np.array(ds.current_cases.iloc[idx.cpu().numpy(), :]['DRUG'])

        molecular_input_long = pd.melt(
            molecular_input,
            id_vars=[
                'DRUG',
                'cell_line'],
            var_name='molecular_names',
            value_name='expression')

        result_long = LRP_scores_long.merge(molecular_input_long, how='left')

        result_long.to_csv(
            PATH + 'LRP_' + str(fold) + '.csv',
            mode='w' if it == 0 else 'a',
            header=it == 0)
        it += 1


def np2pd_specific_genes(nparray, current_cell, current_drug, molecular_names, specific_genes):
    tensor = nparray
    ntensor_features = tensor.shape[1]

    unexpanded_tensor = tensor.copy()

    frame = pd.DataFrame(unexpanded_tensor, columns=molecular_names)
    frame = frame[specific_genes]

    frame['DRUG'] = np.array(current_drug)
    frame['cell_line'] = np.array(current_cell)
    long_frame = pd.melt(
        frame,
        id_vars=[
            'DRUG',
            'cell_line'],
        var_name='molecular_names',
        value_name='LRP')

    return long_frame


def calculate_one_side_LRP_specific_genes(
        model,
        ds,
        specific_genes,
        output_feature=0,
        PATH='./results/LRP_specific_genes/',
        fold=None):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    device = tc.device('cuda:0')
    # ds.generate_train_or_test_cases('test')
    model.eval().to(device)

    assert ds.mode == 'test', 'LRP on ' + ds.mode + ' dataset is not possible'

    dl = DataLoader(ds, batch_size=10, shuffle=False, num_workers=8, pin_memory=True)

    all_results = []

    for drug, molecular, outcome, idx in tqdm(dl):
        drug, molecular, outcome = drug.to(device), molecular.to(device), outcome.to(device)

        drug_latent = model.nn1.forward(drug)
        molecular_latent = model.nn2.forward(molecular)
        # print(model.nn2.layers[1].A_dict)

        s = drug_latent.shape
        helper_network = LRP_Linear(s[0] * s[1], s[0] * s[1]).to(device).eval()

        helper_network.linear.weight = nn.Parameter(drug_latent.detach().view(
            s[0] * s[1]) * tc.eye(s[0] * s[1], device=device), requires_grad=True)
        helper_network.linear.bias = nn.Parameter(
            tc.zeros_like(helper_network.linear.bias), requires_grad=True)

        pred = helper_network.forward(molecular_latent.view(s[0] * s[1]))  # .view(s[0],s[1])

        latent_relevance = helper_network.relprop(pred).view(s[0], s[1])

        input_relevance = model.nn2.relprop(latent_relevance).cpu().detach().numpy()

        LRP_scores_long = np2pd_specific_genes(input_relevance,
                                               current_cell=ds.current_cases.iloc[idx.cpu().numpy(),
                                                                                  :]['cell_line'],
                                               current_drug=ds.current_cases.iloc[idx.cpu().numpy(),
                                                                                  :]['DRUG'],
                                               molecular_names=ds.molecular_names,
                                               specific_genes=specific_genes)

        LRP_scores_long = LRP_scores_long.reset_index(drop=True)

        molecular_input = pd.DataFrame(molecular.cpu().numpy(), columns=ds.molecular_names)
        molecular_input = molecular_input[specific_genes]
        molecular_input['cell_line'] = np.array(
            ds.current_cases.iloc[idx.cpu().numpy(), :]['cell_line'])
        molecular_input['DRUG'] = np.array(ds.current_cases.iloc[idx.cpu().numpy(), :]['DRUG'])

        molecular_input_long = pd.melt(
            molecular_input,
            id_vars=[
                'DRUG',
                'cell_line'],
            var_name='molecular_names',
            value_name='expression')

        result_long = LRP_scores_long.merge(molecular_input_long, how='right')
        all_results.append(result_long)

    pd.concat(all_results, axis=0).to_csv(PATH + 'LRP_specific_genes' + str(fold) + '.csv')
