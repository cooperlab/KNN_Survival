# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 22:21:44 2017

@author: mohamed
"""

import numpy as np

#%%============================================================================
# Generate data
#==============================================================================

n, d = (100, 75)

# method inputs
censored = np.random.randint(0, 2, n)
OPTIM_RATIO=0.2
OPTIM_TRAIN_RATIO=0.5
K=5
SHUFFLES=5


#%%============================================================================
#  Getting split indices (unbalanced)
#==============================================================================

def getSplitIdxs(N, OFFSET = 0, 
                 OPTIM_RATIO=0.2, OPTIM_TRAIN_RATIO=0.5,
                 K=5, SHUFFLES=5):
    '''
    Args: input matrix, ratio allocated to optimization, K (no of folds)
    Out: indices of different sets
    '''

    Idxs = {}
    
    idx_all = np.arange(N) + OFFSET
    np.random.shuffle(idx_all)
    
    if OPTIM_RATIO > 0:
        #
        # indices of hyperpar optimization set
        #
        N_optim = int(OPTIM_RATIO * N)
        N_optim_train = int(OPTIM_TRAIN_RATIO * N_optim)
        
        idx_optim_train = list(idx_all[0:N_optim_train])
        idx_optim_valid = list(idx_all[N_optim_train:N_optim])
        Idxs['idx_optim_train'] = idx_optim_train
        Idxs['idx_optim_valid'] = idx_optim_valid
        
        # Remove optimization indices from full list
        idx_all = idx_all[N_optim:]
    
    #
    # indices of cross validation for each fold
    #
    N_cv = N - int(OPTIM_RATIO * N)
                   
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

    #    
    # K-fold cross-validation with shuffles
    #
    for S in range(SHUFFLES):
        
        ThisIdxList = idx_shuffles[S]
        
        # Cycle through folds and get indices
        for k in(range(K)):
            fold_cv_test[S * K + k] = list(ThisIdxList[fold_bounds[k] : fold_bounds[k+1]])
            fold_cv_train[S * K + k] = [j for j in ThisIdxList \
                         if j not in fold_cv_test[S * K + k]]
            # Append optimization (OTHER train + validation) set to training set
            if OPTIM_RATIO > 0:
                fold_cv_train[S * K + k] += idx_optim_train + idx_optim_valid

    Idxs['fold_cv_train'] = fold_cv_train
    Idxs['fold_cv_test'] = fold_cv_test
            
    return Idxs


#%%============================================================================
# Balanced splitting
#==============================================================================

def get_balanced_SplitIdxs(categories, OFFSET = 0, 
                           OPTIM_RATIO=0.2, OPTIM_TRAIN_RATIO=0.5,
                           K=5, SHUFFLES=5):
                               
    """
    Gets split indices with a balanced representation of the various 
    categories in different folds. 'categories' is an array of 
    numeric category indices (eg. 1 == censored patient)
    """
    # sort categories while keeping track of original index
    idx = np.arange(len(categories))
    categories = np.concatenate((idx[:, None], categories[:, None]), axis = 1)
    categories = categories[categories[:,1].argsort()]
    
    def _get_category_SplitIdx(categories, category_identifier):
        
        '''
        Gets list of split indices for a single category
        Assumes categories are sorted.
        '''
        
        # isolate category and get its offset
        category = categories[:, 1] == category_identifier
        offset = list(np.cumsum(category)).index(1)
        N_categ = np.sum(category)
        
        # Get optimization set and K-fold CV indices
        SplitIdxs_thiscateg = \
            getSplitIdxs(N_categ, OFFSET = offset, 
                         OPTIM_RATIO=OPTIM_RATIO,
                         OPTIM_TRAIN_RATIO=OPTIM_TRAIN_RATIO,
                         K=K, SHUFFLES=SHUFFLES)
                                         
        return SplitIdxs_thiscateg
        
    # intialize with first category
    unique_categories = np.unique(categories[:, 1])
    category_identifier = unique_categories[0]
    SplitIdxs = _get_category_SplitIdx(categories, category_identifier)
    
    # Cycle through categories and add their split indices
    for c in range(1, len(unique_categories)):
        
        # get split indices for this category
        category_identifier = unique_categories[c]
        SplitIdxs_thiscateg = _get_category_SplitIdx(categories, category_identifier)
        
        # merge with existing lists of indices
        SplitIdxs['idx_optim_train'] += SplitIdxs_thiscateg['idx_optim_train']
        SplitIdxs['idx_optim_valid'] += SplitIdxs_thiscateg['idx_optim_valid']
        for k in range(K * SHUFFLES):
            SplitIdxs['fold_cv_train'][k] += SplitIdxs_thiscateg['fold_cv_train'][k]
            SplitIdxs['fold_cv_test'][k] += SplitIdxs_thiscateg['fold_cv_test'][k]
    
    return SplitIdxs