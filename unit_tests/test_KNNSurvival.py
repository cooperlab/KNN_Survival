# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:54:33 2017

@author: mohamed
"""

import sys
#sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

from scipy.io import loadmat
import numpy as np

import DataManagement as dm
import KNNSurvival as knn

#%%========================================================================
# Prepare inputs
#==========================================================================

print("Loading and preprocessing data.")

# Load data

#projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
projectPath = "/home/mtageld/Desktop/KNN_Survival/"

dpath = projectPath + "Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
#dpath = projectPath + "Data/SingleCancerDatasets/GBMLGG/Brain_Gene.mat"
#dpath = projectPath + "Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat"

Data = loadmat(dpath)

Features = np.float32(Data['Integ_X'])
#Features = np.float32(Data['Gene_X'])

N, D = Features.shape

if np.min(Data['Survival']) < 0:
    Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1

Survival = np.int32(Data['Survival']).reshape([N,])
Censored = np.int32(Data['Censored']).reshape([N,])
fnames = Data['Integ_Symbs']
#fnames = Data['Gene_Symbs']

RESULTPATH = projectPath + "Results/tmp/"
MONITOR_STEP = 10
description = "GBMLGG_Gene_"

# remove zero-variance features
fvars = np.std(Features, 0)
keep = fvars > 0
Features = Features[:, keep]
fnames = fnames[keep]

# Get split indices - entire cohort
K_OPTIM = 2
K = 3
SHUFFLES = 10
splitIdxs = dm.get_balanced_SplitIdxs(Censored, \
                                      K = K,\
                                      SHUFFLES = SHUFFLES,\
                                      USE_OPTIM = True,\
                                      K_OPTIM = K_OPTIM)


#raise Exception("On purpose.")

#%%============================================================================
# Tune
#==============================================================================

# Instantiate a KNN survival model.
knnmodel = knn.SurvivalKNN(RESULTPATH, description = description)


# Get optimal K and accuracies

CIs = np.zeros([K * SHUFFLES, K_OPTIM])
K_optim = np.zeros([K_OPTIM])

for outer_fold in range(K_OPTIM):

    print("\nOuter fold {} of {}".format(outer_fold, K_OPTIM-1))
    
    # Get model accuracy (includes param tuning)
    tune_params = {'kcv': 5,
                   'shuffles': 1,
                   'Ks': list(np.arange(10, 160, 10)),
                   }
    ci, k_optim = knnmodel.cv_accuracy(Features, Survival, Censored, \
                                       splitIdxs, outer_fold = outer_fold,\
                                       tune_params = tune_params)
    
    CIs[:, outer_fold] = ci
    K_optim[outer_fold] = k_optim

print("\nOptimal Ks: {}".format(K_optim))
print("25th percentile = {}".format(np.percentile(CIs, 25)))
print("50th percentile = {}".format(np.percentile(CIs, 50)))
print("75th percentile = {}".format(np.percentile(CIs, 75)))

