# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:54:33 2017

@author: mohamed
"""

import os
import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
#sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

import _pickle
from scipy.io import loadmat
import numpy as np

import NCA_model as nca
import KNNSurvival as knn


def get_cv_accuracy(dpath, site, dtype, description,
                    RESULTPATH,
                    k_tune_params={},
                    knn_params={},
                    USE_NCA=False,
                    graphParams={},
                    nca_train_params={},
                    n_feats_kcv_params={},
                    bagging_params={}):

    
    """
    Get KNN cross validation accuracy with or without NCA
    """
    
    # Get a dict of function params and save
    params_all = locals()
    with open(RESULTPATH + description + \
                      'params_all.pkl','wb') as f:
        _pickle.dump(params_all, f)
    
    
    # Load data
    #==========================================================================
    
    print("\n--------------------------------------")
    print("Getting cv accuracy: {}, {}".format(site, dtype))
    print("--------------------------------------\n")
    
    print("Loading data.")
    
    Data = loadmat(dpath)
    
    Features = Data[dtype + '_X']
    fnames = Data[dtype + '_Symbs']
    
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
    
    # Instantiate a KNN survival model.
    knnmodel = knn.SurvivalKNN(RESULTPATH_KNN, description=description)
    
    #
    # initialize
    #
    
    n_outer_folds = len(splitIdxs['idx_optim'])
    n_folds = len(splitIdxs['fold_cv_test'][0])
    
    CIs = np.zeros([n_folds, n_outer_folds])
    
    #
    # itirate through folds
    #
    
    #outer_fold = 0
    for outer_fold in range(n_outer_folds):
        
        print("\nOuter fold {} of {}\n".format(outer_fold, n_outer_folds-1))
        
        # isolate features
        # Note, this is done since they will
        # be modified locally in each outer loop
        X = Features.copy()
        
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
                                      
           ncamodel.train(features = X[optimIdxs, :],\
                          survival = Survival[optimIdxs],\
                          censored = Censored[optimIdxs],\
                          COMPUT_GRAPH_PARAMS = graphParams,\
                          **nca_train_params)
           
           # get feature ranks
           ncamodel.rankFeats(X, fnames, rank_type = "weights")
           ncamodel.rankFeats(X, fnames, rank_type = "stdev")
        
           
           # Transform features according to learned nca model
           #===================================================
           
           print("\nTransforming feats according to learned NCA model.")
           
           # get learned weights
           w = np.load(RESULTPATH_NCA_FOLD + 'model/' + description + 'featWeights.npy')  
           W = np.zeros([len(w), len(w)])
           np.fill_diagonal(W, w)
           
           # transform
           X = np.dot(X, W)
        
           # sort X by absolute feature weights
           sort_idxs = np.flip(np.abs(w).argsort(), axis=0)
           X = X[:, sort_idxs]

           # Get bagged post-NCA accuracy
           #================================================

           ci, _, _ = knnmodel.post_nca_cv_accuracy(\
                   X, Survival, Censored,
                   splitIdxs=splitIdxs,
                   outer_fold=outer_fold,
                   k_tune_params=k_tune_params,
                   n_feats_kcv_params=n_feats_kcv_params,
                   bagging_params=bagging_params)

        else:
        
            # Get accuracy without NCA
            ci, _ = knnmodel.cv_accuracy(X, Survival, Censored, 
                                         splitIdxs, outer_fold=outer_fold,
                                         k_tune_params=k_tune_params)
        
        # record result
        CIs[:, outer_fold] = ci
           
        
    print("\nAccuracy")
    print("------------------------")
    print("25th percentile = {}".format(np.percentile(CIs, 25)))
    print("50th percentile = {}".format(np.percentile(CIs, 50)))
    print("75th percentile = {}".format(np.percentile(CIs, 75)))
    
    
    # Save results
    print("\nSaving final results.")
    with open(RESULTPATH + description + 'testing_Ci.txt','wb') as f:
        np.savetxt(f, CIs, fmt='%s', delimiter='\t')


# apply to various datasets
#=============================================================

if __name__ == '__main__':

    # Params
    #=================================================================
    
    # paths ----------------------------------------------------------
    
    #projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
    projectPath = "/home/mtageld/Desktop/KNN_Survival/"
    RESULTPATH_BASE = projectPath + "Results/2_24Sep2017/"
    
    # dataset and description
    sites = ["MM", "GBMLGG", "BRCA", "KIPAN",]
    dtypes = ["Integ", "Gene"]
    
    norm = 2
    Methods = ['cumulative-time', 'non-cumulative']
    
    # KNN params ----------------------------------------------------
    
    k_tune_params = {'kcv': 4,
                     'shuffles': 3,
                     'Ks': list(np.arange(10, 160, 10)),
                     'norm': norm,
                    }
    
    knn_params = {'norm': norm,
                  }
    
    # NCA params  ---------------------------------------------------
    
    graphParams = \
            {'ALPHA': 0.5,
            'LAMBDA': 0,
            'KAPPA': 1.0,
            'OPTIM': 'Adam',
            'LEARN_RATE': 0.01,
            }
    
    nca_train_params = \
            {'PLOT_STEP': 200,
            'MODEL_SAVE_STEP': 200,
            'MAX_ITIR': 50,
            }
    
    n_feats_kcv_params = \
            {'kcv': 4,
             'shuffles': 2,
             'n_feats_max': 150,
             'norm': norm,
             }
    
    bagging_params = \
            {'min_n_feats': 10,
             'n_subspaces': 20,
             'norm': norm,
             }
    
    # Now run experiment
    #=================================================================
    
    for USE_NCA in [False, True]:
        for Method in Methods:
    
            # pass params to dicts
            k_tune_params['Method'] = Method
            knn_params['Method'] = Method
            n_feats_kcv_params['Method'] = Method
            bagging_params['Method'] = Method
            
            # Itirate through datasets
            
            RESULTPATH = RESULTPATH_BASE + \
                         Method + "_" + \
                         str(USE_NCA) + "NCA/"
            success = os.system("mkdir " + RESULTPATH)
            
            if success != 0:
                print("Folder exists, experiment already done.")
                continue
            
            for dtype in dtypes:
                for site in sites:

                    if dtype == "Gene":
                        nca_train_params['BATCH_SIZE'] = 40
                    else:
                        nca_train_params['BATCH_SIZE'] = 200

                    if (site == "MM") and (dtype == "Integ"):
                        continue

                    
                    description = site +"_"+ dtype +"_"
                    dpath = projectPath + "Data/SingleCancerDatasets/"+ site+"/"+ \
                            site +"_"+ dtype+"_Preprocessed.mat"
                    
                    # create output directory
                    os.system('mkdir ' + RESULTPATH + description)
            
                    # get cv accuracy
                    get_cv_accuracy(dpath=dpath, site=site, dtype=dtype,
                                    description=description,
                                    RESULTPATH=RESULTPATH,
                                    k_tune_params=k_tune_params,
                                    knn_params=knn_params,
                                    USE_NCA=USE_NCA,
                                    graphParams=graphParams,
                                    nca_train_params=nca_train_params,
                                    n_feats_kcv_params=n_feats_kcv_params,
                                    bagging_params=bagging_params)
                
