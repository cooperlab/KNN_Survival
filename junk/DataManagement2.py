# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 22:21:44 2017

@author: mohamed

Tools for managing data
"""

import numpy as np


#%%============================================================================
#  Getting split indices (unbalanced)
#==============================================================================

def getSplitIdxs(N, OFFSET = 0, 
                 K = 3, SHUFFLES = 10,
                 USE_OPTIM = True, K_OPTIM = 2):
    '''
    Get split indices for a given set

    Args:
    -----
    N - no of samples
    OFFSET - index offset
    K - K-fold cross validation for training/testing
    SHUFFLES - no of shuffles
    USE_OPTIM - whether or not to use an optimization set
    K_OPTIM - K-fold for division into optimization and other

    Returns:
    --------
    Idxs - indices for various sets
    '''
    
    Idxs = {}
    
    idx_all = np.arange(N) + OFFSET
    np.random.shuffle(idx_all)
    
    def get_cv_idxs(idxs, kcv, n_shuffles):
    
        """Get K-fold cross validation indices"""
    
        #
        # indices of cross validation for each fold
        #
    
        N_cv = len(idxs)
                       
        fold_bounds = np.arange(0, N_cv, int(N_cv / kcv))
        fold_bounds = list(fold_bounds[0:kcv])
        fold_bounds.append(N_cv-1)
        
        fold_bounds = np.int64(fold_bounds)
        
        fold_cv_test = list(np.zeros(kcv * n_shuffles))
        fold_cv_train = list(np.zeros(kcv * n_shuffles))
        
        
        # Doing all the shufling first since for some reason
        # np shuffle does not work insider the next loop!
        idx_shuffles = list(np.zeros(n_shuffles))
        for shuff in range(n_shuffles):
            np.random.shuffle(idxs)
            idx_shuffles[shuff] = idxs.copy()
    
        #    
        # K-fold cross-validation with shuffles
        #
        for S in range(n_shuffles):
            
            ThisIdxList = idx_shuffles[S]
            
            # Cycle through folds and get indices
            for k in(range(kcv)):
                fold_cv_test[S * kcv + k] = \
                    list(ThisIdxList[fold_bounds[k] : fold_bounds[k+1]])
                fold_cv_train[S * kcv + k] = \
                    [j for j in ThisIdxList if j not in fold_cv_test[S * kcv + k]]
    
        return fold_cv_train, fold_cv_test
    
    
    # Get assignment into optimization and other
    if USE_OPTIM:
        other_idxs, optimization_idxs = \
            get_cv_idxs(idx_all, K_OPTIM, n_shuffles=1)
        n_folds = len(other_idxs)
    else:
        n_folds = 1
    
    # itirate through various folds and get KCV indices
    fold_cv_train = []
    fold_cv_test = []
        
    for fold_no in range(n_folds):    
        
        if USE_OPTIM:        
            
            train, test = \
                get_cv_idxs(other_idxs[fold_no], K, n_shuffles=SHUFFLES)
            
            # append optimization indices to training set
            for f in range(len(train)):
                train[f] += optimization_idxs[fold_no]
        else:
            train, test = \
                get_cv_idxs(idx_all, K, n_shuffles=SHUFFLES)   
    
        # hold final indices
        fold_cv_train.append(train)
        fold_cv_test.append(test)
    
    # wrap up and save results
    if USE_OPTIM:
        Idxs['idx_optim'] = optimization_idxs
    Idxs['fold_cv_train'] = fold_cv_train
    Idxs['fold_cv_test'] = fold_cv_test
    
    return(Idxs)

#%%============================================================================
# Balanced splitting
#==============================================================================

def get_balanced_SplitIdxs(categories,
                           K=3, SHUFFLES=10,
                           USE_OPTIM = True,
                           K_OPTIM = 2):
                               
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
                         K = K, SHUFFLES = SHUFFLES,
                         USE_OPTIM = USE_OPTIM, K_OPTIM = K_OPTIM)
                                         
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
        for fold in range(K_OPTIM):
            if USE_OPTIM > 0:
                SplitIdxs['idx_optim'][fold] += SplitIdxs_thiscateg['idx_optim'][fold]

            for k in range(K * SHUFFLES):
                SplitIdxs['fold_cv_train'][fold][k] += SplitIdxs_thiscateg['fold_cv_train'][fold][k]
                SplitIdxs['fold_cv_test'][fold][k] += SplitIdxs_thiscateg['fold_cv_test'][fold][k]
        
    return SplitIdxs

 
#%%============================================================================
# Balanced batches
#==============================================================================

def get_balanced_batches(categories, BATCH_SIZE):
    
    """
    Gets indices for various batches (to be used for stochastic GD)
    in a balanced fashion - i.e. so that the batches have nearly equal
    representations of each category/property.
    """
    K = int(categories.shape[0] / BATCH_SIZE)
    batchIdxs = get_balanced_SplitIdxs(categories, OPTIM_RATIO=0, K=K, SHUFFLES=1)
    batchIdxs = batchIdxs['fold_cv_test'][0:K]
    
    return batchIdxs
    
    
#%%############################################################################
#%%############################################################################
#%%############################################################################
#%%############################################################################

#%%============================================================================
# test methods
#==============================================================================

if __name__ == '__main__':
    
    # Load data
    N = 500
    OFFSET = 0
    K = 3
    SHUFFLES = 10
    USE_OPTIM = True
    K_OPTIM = 2

    # method inputs
    censored = np.random.binomial(1, 0.2, N)
    
   # # get split indices
   # Idxs = getSplitIdxs(N, OFFSET = OFFSET, 
   #                     K = K, SHUFFLES = SHUFFLES,
   #                     USE_OPTIM = USE_OPTIM, 
   #                     K_OPTIM = K_OPTIM)

    # get balances idxs
    Idxs = get_balanced_SplitIdxs(censored, 
                                  K = K, SHUFFLES = SHUFFLES,
                                  USE_OPTIM = USE_OPTIM, K_OPTIM = K_OPTIM)
