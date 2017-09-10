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
import NCA_model as nca

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
#splitIdxs = dm.get_balanced_SplitIdxs(Censored)


#%%============================================================================
# Train
#==============================================================================
ncamodel = nca.SurvivalNCA(RESULTPATH, description = description)#, \
#                           LOADPATH = RESULTPATH + 'model/' + description + \
#                           'ModelAttributes.pkl')
ncamodel.train(features = Features,
             survival = Survival,
             censored = Censored,
             BATCH_SIZE = 50)
             
ncamodel.rankFeats(Features, fnames, rank_type = "weights")


#W = np.load(ncamodel.RESULTPATH + 'model/' + ncamodel.description + 'featWeights.npy')  