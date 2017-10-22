# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:32:30 2017

@author: mohamed
"""

import os
import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes/')

import numpy as np
from pandas import DataFrame as df, read_table
from scipy.io import loadmat
import _pickle

import SurvivalUtils as sUtils

#%%
# Read predictions and outcomes
#==============================================================================

base_path = '/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/'
result_path = base_path + 'Results/9_8Oct2017/'

# Initialize final dataframe
#==============================================================================

n_folds = 30
idxNames = ['site', 'dtype']
folds = list(np.arange(n_folds))
folds = ['fold_' + str(j) for j in folds]
idxNames.extend(folds)
idxNames.extend(['median', 'mean', '25%ile', '75%ile', 'stdev'])

CIs = df(index=idxNames)

print("\nsite\tdtype")
print("-------------------------------")

#site = "GBMLGG"
#dtype = "Gene" #"Integ"
for site in ["GBMLGG", "KIPAN"]:
    for dtype in ["Integ", "Gene"]:

        print(site + "\t" + dtype)
        
        # Load outcomes data and split indices 
        #==============================================================================
        
        dpath = base_path + "Data/SingleCancerDatasets/"+ site+"/"+ \
                    site +"_"+ dtype+"_Preprocessed.mat"
        
        # Load outcomes
        Data = loadmat(dpath)
        N = Data[dtype + '_X'].shape[0]
        Survival = Data['Survival'].reshape([N,])
        Censored = Data['Censored'].reshape([N,])
        Data = None
        
        # load split indices
        with open(dpath.split('.mat')[0] + '_splitIdxs.pkl','rb') as f:
                splitIdxs = _pickle.load(f)
        
        # Load file list
        pred_path = result_path + site + '_' + dtype + '/'
        result_files = os.listdir(pred_path)
        val_files = [j for j in result_files if 'preds_val' in j]
        test_files = [j for j in result_files if 'preds_test' in j]
        
        
        # Cycle through folds
        #==============================================================================
        
        ci_test = []
        #fold = 0
        for fold, foldname in enumerate(folds):
            
            print("\t{}".format(foldname))
        
            # Get predictions
            foldidx_val = ['fold_{}_'.format(fold+1) in j for j in val_files].index(True)
            foldidx_test = ['fold_{}_'.format(fold+1) in j for j in test_files].index(True)
            preds_val = read_table(pred_path + val_files[foldidx_val], sep=' ')
            preds_test = read_table(pred_path + test_files[foldidx_test], sep=' ')
            
            # Get validation set accuracy
            ci_val = []
            for hyperpars in range(preds_val.shape[1]):
                ci_val.append(sUtils.c_index(preds_val.values[:, hyperpars], 
                                             Survival[splitIdxs['valid'][fold]], 
                                             Censored[splitIdxs['valid'][fold]], 
                                             prediction_type = 'risk'))
                                             
            # Get testing set accuracy for optimal hyperparams
            ci_test.append(sUtils.c_index(preds_test.values[:, np.argmax(ci_val)], 
                                          Survival[splitIdxs['test'][fold]], 
                                          Censored[splitIdxs['test'][fold]], 
                                          prediction_type = 'risk'))
                                          
        # append summary stats
        ci_test.extend([np.median(ci_test), np.mean(ci_test), \
                        np.percentile(ci_test, 25), np.percentile(ci_test, 75), \
                        np.std(ci_test)])
        
        # append to final results table
        ci_merge = [site, dtype]
        ci_merge.extend(ci_test)
        CIs[CIs.shape[1]] = ci_merge
        
        
# now save
CIs.to_csv(result_path + "results_merged.tab", sep="\t")