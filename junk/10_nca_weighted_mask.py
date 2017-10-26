# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:57:35 2017

@author: mohamed
"""

import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
#sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

#import os
import numpy as np
#import matplotlib.pylab as plt
from scipy.io import loadmat
#from scipy.stats import spearmanr

import SurvivalUtils as sUtils

#%%============================================================================
# Define params
#==============================================================================

base_path = '/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/'
#base_path = '/home/mtageld/Desktop/KNN_Survival/'
result_path = base_path + 'Results/12_21Oct2017/'

sites = ["GBMLGG", "KIPAN"]
dtypes = ["Integ", ] # "Gene"]
methods = ["cumulative-time_TrueNCA_FalsePCA", "non-cumulative_TrueNCA_FalsePCA"]

n_top_folds = 30
pval_thresh = 0.01
n_feats_to_plot = 10

site = sites[0]
dtype = dtypes[0]
method = methods[0]

#%% 
# Get feature files
#==============================================================================

dpath = base_path + "Data/SingleCancerDatasets/"+ site+"/" + \
        site +"_"+ dtype+"_Preprocessed.mat"

print("Loading data.")
Data = loadmat(dpath)
Features = Data[dtype + '_X'].copy()
N = Features.shape[0]
P = Features.shape[1]
Survival = Data['Survival'].reshape([N,])
Censored = Data['Censored'].reshape([N,])
fnames = Data[dtype + '_Symbs']
fnames = [j.split(' ')[0] for j in fnames]
Data = None

#%% 
# Get result files
#==============================================================================

mask_type = 'observed'
# Getting at-risk groups
t_batch, o_batch, at_risk_batch, x_batch = \
    sUtils.calc_at_risk(Survival, 
                        1-Censored,
                        Features)

sys.exit()

#%%                        
# Get mask (to be multiplied by Pij) ******************

n_batch = t_batch.shape[0]
Pij_mask = np.zeros((n_batch, n_batch))

# Get difference in outcomes between all cases
if mask_type == 'observed':
    outcome_diff = np.abs(t_batch[None, :] - t_batch[:, None])
    
for idx in range(n_batch):
    
    # only observed cases
    if o_batch[idx] == 1:
        
        if mask_type == 'at-risk':
            # only at-risk cases (unweighted)
            Pij_mask[idx, at_risk_batch[idx]:] = 1
            
        elif mask_type == 'observed':
            # only observed cases (weighted)
            Pij_mask[idx, o_batch==1] = 1
            
if mask_type == 'observed':
    Pij_mask = Pij_mask * outcome_diff