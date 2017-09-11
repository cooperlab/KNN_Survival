# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:54:33 2017

@author: mohamed
"""

import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')

from scipy.io import loadmat
import numpy as np

import DataManagement as dm
import KNNSurvival as knn

#%%========================================================================
# Prepare inputs
#==========================================================================

print("Loading and preprocessing data.")

# Load data

projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
#projectPath = "/home/mtageld/Desktop/KNN_Survival/"

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

# Get split indices
splitIdxs = dm.get_balanced_SplitIdxs(Censored)

# Get training and testing sets
fold = 10
idxs_train = splitIdxs['fold_cv_train'][fold]
idxs_test = splitIdxs['fold_cv_test'][fold]

X_test = Features[idxs_test, :]
X_train = Features[idxs_train, :]
Survival_train = Survival[idxs_train]
Censored_train = Censored[idxs_train]
Survival_test = Survival[idxs_test]
Censored_test = Censored[idxs_test]

#%%============================================================================
# Train
#==============================================================================

# Instantiate a KNN survival model
knnmodel = knn.SurvivalKNN(RESULTPATH, description = description)

#%%
# Predict testing set
T_test, Ci = knnmodel.predict(X_test, X_train, 
                              Survival_train, Censored_train, 
                              Survival_test = Survival_test, 
                              Censored_test = Censored_test, 
                              K = 80)