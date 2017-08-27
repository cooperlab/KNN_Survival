#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:12:57 2017

@author: mohamed

A group of supporting utilities.
"""

import numpy as np
from scipy.io import loadmat


#%%============================================================================
# GetSplitIdxs
#==============================================================================

def getSplitIdxs(data, OPTIM_RATIO=0.2, K=5, SHUFFLES=5):
    '''
    Args: input matrix, ratio allocated to optimization, K (no of folds)
    Out: indices of different sets
    '''

    Idxs = {}
    
    N_all = np.size(data, 0)
    idx_all = np.arange(N_all)
    data = None
    
    if OPTIM_RATIO > 0:
        # indices of hyperpar optimization set
        N_optim = int(OPTIM_RATIO * N_all)
        idx_optim_train = list(np.arange(0, int(0.5 * N_optim)))
        idx_optim_valid = list(np.arange(int(0.5 * N_optim), N_optim))
        Idxs['idx_optim_train'] = idx_optim_train
        Idxs['idx_optim_valid'] = idx_optim_valid
        
        # Remove optimization indices from full list
        idx_all = idx_all[N_optim:N_all]
    
    # indices of cross validation for each fold
    N_cv = N_all - int(OPTIM_RATIO * N_all)
                   
    fold_bounds = np.arange(0, N_cv, int(N_cv / K))
    fold_bounds = list(fold_bounds[0:K])
    fold_bounds.append(N_cv-1)
    
    fold_bounds = np.int64(fold_bounds)
    
    fold_cv_test = list(np.zeros(K * SHUFFLES))
    fold_cv_train = list(np.zeros(K * SHUFFLES))
    
    
    # Doing all the shufling first since for some reason
    # np shuffle does not work insider the next loop!
    idx_shuffles = list(np.zeros(SHUFFLES))
    for shuff in range(SHUFFLES):
        np.random.shuffle(idx_all)
        idx_shuffles[shuff] = idx_all.copy()

    # K-fold cross-validation with shuffles
    for S in range(SHUFFLES):
        
        ThisIdxList = idx_shuffles[S]
        
        # Cycle through folds and get indices
        for k in(range(K)):
            fold_cv_test[S * K + k] = ThisIdxList[fold_bounds[k] : fold_bounds[k+1]]
            fold_cv_train[S * K + k] = [j for j in ThisIdxList \
                         if j not in fold_cv_test[S * K + k]]
            # Append optimization (OTHER train + validation) set to training set
            if OPTIM_RATIO > 0:
                fold_cv_train[S * K + k] = fold_cv_train[S * K + k] +\
                                              idx_optim_train + idx_optim_valid

    Idxs['fold_cv_train'] = fold_cv_train
    Idxs['fold_cv_test'] = fold_cv_test
            
    return Idxs


#%%============================================================================
# timeIndicator
#==============================================================================

def getAliveStatus(Survival, Censored, t_min = 0, t_max = 0, scale = 1):

    """ 
    This converts survival and censored data (where 1 = lost to follow-up)
    into alive-dead data by adding a time indicator variable 
    (where 1 = alive, 0 = dead, -1 = unknown survival status).
    Use scale to convert time scale (eg to convert from days to weeks
    set scale = 7)
    """
    
    # Get data in needed scale
    Survival = np.int32(np.floor(Survival / scale))
    
    if t_max == 0:
        t_max = np.max(Survival)
    
    # Initialize output matrix - rows are patients, cols are time points
    aliveStatus = np.ones([len(Survival), t_max - t_min +1])
    
    #idx = 0
    for idx in range(len(Survival)):
    
        if Censored[idx,0] == 0:
            # known death time
            aliveStatus[idx,Survival[idx,0]+1:] = 0 
        else:
            # lost to follow-up
            aliveStatus[idx,Survival[idx,0]+1:] = -1
            
    return aliveStatus


#%%============================================================================
# test methods
#==============================================================================

if __name__ == '__main__':
    
    # Load data
    dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
    Data = loadmat(dpath)
    
    data = np.float32(Data['Integ_X'])
    if np.min(Data['Survival']) < 0:
        Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1
    
    Survival = np.int32(Data['Survival'])
    Censored = np.int32(Data['Censored'])
    
    # Get split indices
    splitIdxs = getSplitIdxs(data)
    
    # Generate survival status - discretized into months
    aliveStatus = getAliveStatus(Survival, Censored, scale = 30)
    