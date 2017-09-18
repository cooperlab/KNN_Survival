# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:54:33 2017

@author: mohamed
"""

import os
import sys
#sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

import _pickle
from scipy.io import loadmat
import numpy as np

import NCA_model as nca
import KNNSurvival as knn

def get_cv_accuracy(dpath, site, dtype, description,\
                    RESULTPATH, \
                    k_tune_params = {}, knn_params = {}, \
                    USE_NCA = False, \
                    graphParams = {}, nca_train_params = {}):
    """
    Get KNN cross validation accuracy with or without NCA
    """
    
    # Load data
    #==========================================================================
    
    print("\n--------------------------------------")
    print("Getting cv accuracy: {}, {}".format(site, dtype))
    print("--------------------------------------\n")

    print("Loading data.")
    
    Data = loadmat(dpath)
    
    if dtype == 'Integ':
        Features = Data['Integ_X']
        fnames = Data['Integ_Symbs']
    else:
        Features = Data['Gene_X']
        fnames = Data['Gene_Symbs']
    
    N = Features.shape[0]
    Survival = Data['Survival'].reshape([N,])
    Censored = Data['Censored'].reshape([N,])

    with open(dpath.split('.mat')[0] + '_splitIdxs.pkl','rb') as f:
        splitIdxs = _pickle.load(f)

    #
    # result structure
    #
    
    RESULTPATH_NCA = RESULTPATH + "nca/"
    RESULTPATH_KNN = RESULTPATH + "knn/"
    LOADPATH = None
    
    os.system('mkdir ' + RESULTPATH_NCA)
    os.system('mkdir ' + RESULTPATH_KNN)
    
    # Go through outer folds, optimize and get accuracy
    #==============================================================================
    
    #
    # initialize
    #
    
    n_outer_folds = len(splitIdxs['idx_optim'])
    n_folds = len(splitIdxs['fold_cv_test'][0])
    
    CIs = np.zeros([n_folds, n_outer_folds])
    K_optim = np.zeros([n_outer_folds])
    
    #
    # itirate through folds
    #
    
    for outer_fold in range(n_outer_folds):
    
        print("\nOuter fold {} of {}\n".format(outer_fold, n_outer_folds-1))
    
        # Isolate optimization set 
        optimIdxs = splitIdxs['idx_optim'][outer_fold]
    
        if USE_NCA:
    
           # Learn NCA matrix on optimization set
           #========================================
           
           print("\nLearning NCA on optimization set\n")
           
           # instantiate
    
           RESULTPATH_NCA_FOLD = RESULTPATH_NCA + "fold_{}/".format(outer_fold)
           os.system("mkdir " + RESULTPATH_NCA_FOLD)
    
           ncamodel = nca.SurvivalNCA(RESULTPATH_NCA_FOLD, \
                                      description = description, \
                                      LOADPATH = LOADPATH)
                                      
           ncamodel.train(features = Features[optimIdxs, :],\
                          survival = Survival[optimIdxs],\
                          censored = Censored[optimIdxs],\
                          COMPUT_GRAPH_PARAMS = graphParams,\
                          **nca_train_params)
           
           # get feature ranks
           ncamodel.rankFeats(Features, fnames, rank_type = "weights")
           ncamodel.rankFeats(Features, fnames, rank_type = "stdev")
    
           
           # Transform features according to learned nca model
           #===================================================
           
           print("\nTransforming feats according to learned NCA model.")
           
           # get learned weights
           w = np.load(RESULTPATH_NCA_FOLD + 'model/' + description + 'featWeights.npy')  
           W = np.zeros([len(w), len(w)])
           np.fill_diagonal(W, w)
           
           # transform
           Features = np.dot(Features, W)
       
    
        # Get model accuracy
        #=====================
        
        # Instantiate a KNN survival model.
        knnmodel = knn.SurvivalKNN(RESULTPATH_KNN, description = description)
        
        
        # get accuracy
        print("\nGetting accuracy.")
        
        ci, k_optim = knnmodel.cv_accuracy(Features, Survival, Censored, \
                                           splitIdxs, outer_fold = outer_fold,\
                                           tune_params = k_tune_params)
        CIs[:, outer_fold] = ci
        K_optim[outer_fold] = k_optim
        
        
        
    print("\nAccuracy")
    print("------------------------")
    print("25th percentile = {}".format(np.percentile(CIs, 25)))
    print("50th percentile = {}".format(np.percentile(CIs, 50)))
    print("75th percentile = {}".format(np.percentile(CIs, 75)))
    
    
    # Save results
    Results = {'USE_NCA': USE_NCA,
               'graphParams': graphParams,
               'nca_train_params': nca_train_params,
               'k_tune_params': k_tune_params,
               'knn_params': knn_params,
               'CIs': CIs,
               'K_optim': K_optim,
               }
    
    
    print("\nSaving final results.")
    with open(RESULTPATH + description + \
                      'Results.pkl','wb') as f:
        _pickle.dump(Results, f)



# apply to various datasets
#=================================================================

if __name__ == '__main__':
    
    # Params
    #=================================================================
    
    # paths
    #projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
    projectPath = "/home/mtageld/Desktop/KNN_Survival/"
    RESULTPATH = projectPath + "Results/tmp/"
    
    # dataset and description
    sites = ["GBMLGG", "BRCA", "KIPAN",]# "LUSC"]
    dtypes = ["Integ",] # "Gene"]
    
    # KNN params
    norm = 2
    Method = 'cumulative_hazard'
    
    k_tune_params = {'kcv': 4,
                     'shuffles': 5,
                     'Ks': list(np.arange(10, 160, 10)),
                     'norm': norm,
                     'Method': Method,
                    }
    
    knn_params = {'norm': norm,
                  'Method': Method,
                  }
    
    
    # NCA params
    USE_NCA = True
    graphParams = {'ALPHA': 0.5,
                   'LAMBDA': 0,
                   'KAPPA': 1.0,
                   'OPTIM': 'GD',
                   'LEARN_RATE': 0.01}
    
    nca_train_params = {'BATCH_SIZE': 200,
                        'PLOT_STEP': 200,
                        'MODEL_SAVE_STEP': 200,
                        'MAX_ITIR': 100,
                       }
    
    
    # Itirate through datasets
    #=================================================================

    for dtype in dtypes:
        for site in sites:

            description = site +"_"+ dtype +"_"
            dpath = projectPath + "Data/SingleCancerDatasets/"+ site+"/"+ site +"_"+ dtype+"_Preprocessed.mat"

            # create output directory
            os.system('mkdir ' + RESULTPATH + description)

            # now get accuracy and save
            get_cv_accuracy(dpath=dpath, site=site, dtype=dtype, \
                            description = description,\
                            RESULTPATH = RESULTPATH + description + '/', \
                            k_tune_params = k_tune_params, \
                            knn_params = knn_params, \
                            USE_NCA = USE_NCA, \
                            graphParams = graphParams, \
                            nca_train_params = nca_train_params)

