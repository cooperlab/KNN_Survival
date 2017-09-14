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
splitIdxs = dm.get_balanced_SplitIdxs(Censored, \
                                      K = 3,\
                                      SHUFFLES = 10,\
                                      USE_OPTIM = True,\
                                      K_OPTIM = 2)


#raise Exception("On purpose.")

#%%============================================================================
# Tune
#==============================================================================

# Instantiate a KNN survival model.
knnmodel = knn.SurvivalKNN(RESULTPATH, description = description)

fold = 0

# Get optimization set
optimIdxs = splitIdxs['idx_optim'][fold]

# Get optimal K using optimization set
CIs_K, K_optim = knnmodel.cv_tune(Features[optimIdxs, :], \
                                  Survival[optimIdxs], \
                                  Censored[optimIdxs], \
                                  kcv = 5, \
                                  shuffles = 5, \
                                  Ks = list(np.arange(10, 160, 10)))


# Get model accuracy
#CIs = knnmodel.cv_accuracy(Features, Survival, Censored, \
#                           splitIdxs, K = 80)


