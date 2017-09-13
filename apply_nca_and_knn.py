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
import NCA_model as nca
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

# remove zero-variance features
fvars = np.std(Features, 0)
keep = fvars > 0
Features = Features[:, keep]
fnames = fnames[keep]

# params
RESULTPATH = projectPath + "Results/tmp/"
MONITOR_STEP = 10
description = "GBMLGG_Integ_"
LOADPATH = None
#LOADPATH = RESULTPATH + 'model/' + description + 'ModelAttributes.pkl'

#%%========================================================================
# Get split indices
#==========================================================================

# Get split indices - entire cohort
splitIdxs = dm.get_balanced_SplitIdxs(Censored, OPTIM_RATIO = 0.5,\
                                      K = 3,\
                                      SHUFFLES = 10)

# Isolate optimization set 
optimIdxs = splitIdxs['idx_optim_train'] + splitIdxs['idx_optim_valid']


#%%============================================================================
# Learn NCA matrix on optimization set
#==============================================================================

# Instantiate
ncamodel = nca.SurvivalNCA(RESULTPATH, description = description, \
                           LOADPATH = LOADPATH)

# train
graphParams = {'ALPHA': 0.5,
               'LAMBDA': 0, 
               'OPTIM': 'GD',
               'LEARN_RATE': 0.01}
                           
ncamodel.train(features = Features[optimIdxs, :],
               survival = Survival[optimIdxs],
               censored = Censored[optimIdxs],
               COMPUT_GRAPH_PARAMS = graphParams,
               BATCH_SIZE = 200,
               MAX_ITIR = 100)

# get feature ranks
ncamodel.rankFeats(Features, fnames, rank_type = "weights")
ncamodel.rankFeats(Features, fnames, rank_type = "stdev")


#%%============================================================================
# Transform features according to learned nca model
#==============================================================================

# get learned weights
w = np.load(RESULTPATH + 'model/' + description + 'featWeights.npy')  
W = np.zeros([w, w])
np.fill_diagonal(W, w)

# transform
Features = np.dot(Features, W)
