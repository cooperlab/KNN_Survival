#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 01:56:26 2017

@author: mohamed

Survival NCA (Neighborhood Component Analysis)
"""

# Append relevant paths
import os
import sys

def conditionalAppend(Dir):
    """ Append dir to sys path"""
    if Dir not in sys.path:
        sys.path.append(Dir)

cwd = os.getcwd()
conditionalAppend(cwd)

from scipy.io import loadmat, savemat
import numpy as np
import SurvivalUtils as sUtils
import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.INFO)


#raise Exception()

#%%============================================================================
# ---- J U N K ----------------------------------------------------------------
#==============================================================================

# Load data
dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
Data = loadmat(dpath)

data = np.float32(Data['Integ_X'])
if np.min(Data['Survival']) < 0:
    Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1

Survival = np.int32(Data['Survival'])
Censored = np.int32(Data['Censored'])
fnames = Data['Integ_Symbs']

# Get split indices
#splitIdxs = sUtils.getSplitIdxs(data)

n = 50
data = data[0:n,:]
Survival = Survival[0:n,:]
Censored = Censored[0:n,:]

# remove zero-variance features
fvars = np.std(data, 0)
toKeep = fvars > 0
data = data[:, toKeep]
fnames = fnames[toKeep]
fvars = fvars[toKeep]

# Generate survival status - discretized into months
aliveStatus = sUtils.getAliveStatus(Survival, Censored, scale = 30)


#%%============================================================================
# --- P R O T O T Y P E S -----------------------------------------------------
#==============================================================================

RESULTPATH = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/tmp/"

LEARN_RATE = 0.05
D_new = data.shape[1] # set D_new < D to reduce dimensions
MONITOR_STEP = 10


#%%============================================================================
# Setting things up
#==============================================================================

# Get dims
N, D = np.int32(data.shape)
T = np.int32(aliveStatus.shape[1]) # no of time points

# Initialize A to a scaling matrix
A_init = np.zeros((D, D_new))

epsilon = 1e-7 # to  avoid division by zero

np.fill_diagonal(A_init, 1./(data.max(axis=0) - data.min(axis=0) + epsilon))
#np.fill_diagonal(A_init, 1./(np.sqrt(fvars) + epsilon))
#np.fill_diagonal(A_init, np.random.rand(D))
#A_init = np.random.rand(D, D_new)

A_init = np.float32(A_init)


#%%============================================================================
# 
#==============================================================================


#%%============================================================================
# Now parse the learned matrix A and save
#==============================================================================

#def getRanks(A):
#    w = np.diag(A).reshape(D_new, 1)
#    fidx = np.arange(len(A)).reshape(D_new, 1)
#    
#    w = np.concatenate((fidx, w), 1)
#    w = w[w[:,1].argsort()][::-1]
#    tokeep = w[:,1] < 100000
#    w = w[tokeep,:]
#    
#    fnames_ranked = fnames[np.int32(w[:,0])]
#    
#    return fnames_ranked
#
#ranks_init = getRanks(A_init)
#ranks_current = getRanks(A_current)
#
## Save analysis result
#result = {'A_init': A_init,
#          'A_current': A_current,
#          'ranks_init': ranks_init,
#          'ranks_current': ranks_current,
#          'LEARN_RATE': LEARN_RATE,}
#
#savemat(RESULTPATH + 'result', result)
