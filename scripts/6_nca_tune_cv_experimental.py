#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:38:22 2017

@author: mtageld
"""

import os
import sys
#sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

import _pickle
from scipy.io import loadmat
import numpy as np

import NCA_model_experimental as nca
import KNNSurvival as knn

#%%

def get_cv_accuracy(dpath, site, dtype, description,
                    RESULTPATH,
                    k_tune_params={},
                    knn_params={},
                    USE_NCA=False,
                    graphParams={},
                    nca_train_params={},
                    n_feats_kcv_params={},
                    bagging_params={},
                    elastic_net_params={}):
    
    """
    Get KNN cross validation accuracy with or without NCA
    """
    
    # Get a dict of function params and save
    params_all = locals()
    with open(RESULTPATH + description + \
                      'params_all.pkl','wb') as f:
        _pickle.dump(params_all, f)

    print("\n--------------------------------------")
    print("Getting cv accuracy: {}, {}".format(site, dtype))
    print("--------------------------------------\n")
    
    print("Loading data.")
    
    Data = loadmat(dpath)
    
    Features = Data[dtype + '_X']
    
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
        
        #USE_NCA = True
        if USE_NCA:
            
            # instantiate NCA model
            ncamodel = nca.SurvivalNCA(RESULTPATH_NCA, \
                                       description = description, \
                                       LOADPATH = LOADPATH)
                                       
            #%%
            # Finding optimal values for ALPHA and LAMBDA (regularization)
            #==============================================================
            
            ALPHAS = np.arange(0, 1.1, 0.2)
            LAMBDAS = np.arange(0, 1.1, 0.2)
            
            # Get training and validation set
            stoppoint = int(elastic_net_params['VALID_RATIO'] * len(optimIdxs))
            optimIdxs_valid = optimIdxs[0:stoppoint]
            optimIdxs_train = optimIdxs[stoppoint:]
            
            x_valid = X[optimIdxs_valid, :]
            
            cis = []
            
            for ALPHA in ALPHAS:
                for LAMBDA in LAMBDAS:
                    
                    if ((LAMBDA == 0) and (ALPHA > ALPHA.min())):
                        continue
            
                    graphParams['ALPHA'] = ALPHA
                    graphParams['LAMBDA'] = LAMBDA
                    
                    #w = ncamodel.train(features = X[optimIdxs_train, :],\
                    #                   survival = Survival[optimIdxs_train],\
                    #                   censored = Censored[optimIdxs_train],\
                    #                   COMPUT_GRAPH_PARAMS = graphParams,\
                    #                   **nca_train_params)
                    W = ncamodel.train(features = X[optimIdxs_train, :],\
                                       survival = Survival[optimIdxs_train],\
                                       censored = Censored[optimIdxs_train],\
                                       COMPUT_GRAPH_PARAMS = graphParams,\
                                       **nca_train_params)
                    ncamodel.reset_TrainHistory()
                    
                    # transform
                    #W = np.zeros((len(w), len(w)))
                    #np.fill_diagonal(W, w)
                    x_valid_transformed = np.dot(X[optimIdxs_valid, :], W)
                    x_train_transformed = np.dot(X[optimIdxs_train, :], W)
                    
                    # get neighbor indices    
                    neighbor_idxs = knnmodel._get_neighbor_idxs(x_valid_transformed, 
                                                                x_train_transformed, 
                                                                norm = norm)
                    
                    # Predict testing set
                    _, Ci = knnmodel.predict(neighbor_idxs,
                                             Survival_train=Survival[optimIdxs_train], 
                                             Censored_train=Censored[optimIdxs_train], 
                                             Survival_test = Survival[optimIdxs_valid], 
                                             Censored_test = Censored[optimIdxs_valid], 
                                             K = elastic_net_params['K'], 
                                             Method = Method)
                    
                    cis.append([ALPHA, LAMBDA, Ci])
                    
                    print("\n----------------------")
                    print("ALPHA\tLAMBDA\tCi")
                    print("{}\t{}\t{}".format(ALPHA, LAMBDA, round(Ci, 3)))
                    print("----------------------\n")
            
            cis = np.array(cis)
            optimal = cis[:,2].argmax()
            ALPHA_OPTIM = cis[optimal, 0]
            LAMBDA_OPTIM = cis[optimal, 1]
            
            print("\nOptimal Alpha, Lambda = {}, {}".format(ALPHA_OPTIM, LAMBDA_OPTIM))
            
            #%%
            
            # Learn NCA matrix on optimization set
            #========================================
            
            print("\nLearning final NCA matrix\n")
            
            graphParams['ALPHA'] = ALPHA_OPTIM
            graphParams['LAMBDA'] = LAMBDA_OPTIM
            
            #w = ncamodel.train(features = X[optimIdxs, :],\
            #                   survival = Survival[optimIdxs],\
            #                   censored = Censored[optimIdxs],\
            #                   COMPUT_GRAPH_PARAMS = graphParams,\
            #                   **nca_train_params)
            W = ncamodel.train(features = X[optimIdxs, :],\
                               survival = Survival[optimIdxs],\
                               censored = Censored[optimIdxs],\
                               COMPUT_GRAPH_PARAMS = graphParams,\
                               **nca_train_params)
            #W = np.zeros((len(w), len(w)))
            #np.fill_diagonal(W, w) 
            
            # Transform features according to learned nca model
            X = np.dot(X, W)
            
            print("\nGetting accuracy.")        
            
            # Get bagged post-NCA accuracy
            #================================================
            
            # sort X by absolute feature weights
            #sort_idxs = np.flip(np.abs(w).argsort(), axis=0)
            #X = X[:, sort_idxs]
            #
            #ci, _, _ = knnmodel.post_nca_cv_accuracy(\
            #        X, Survival, Censored,
            #        splitIdxs=splitIdxs,
            #        outer_fold=outer_fold,
            #        k_tune_params=k_tune_params,
            #        n_feats_kcv_params=n_feats_kcv_params,
            #        bagging_params=bagging_params)
            
            # just get accuracy
            ci, _ = knnmodel.cv_accuracy(X, Survival, Censored, \
                                         splitIdxs, outer_fold=outer_fold,\
                                         k_tune_params=k_tune_params)
        
        else:
            # just get accuracy
            ci, _ = knnmodel.cv_accuracy(X, Survival, Censored, \
                                         splitIdxs, outer_fold=outer_fold,\
                                         k_tune_params=k_tune_params)
        # record result
        CIs[:, outer_fold] = ci
    
    #%%    
    print("\nAccuracy")
    print("------------------------")
    print("25th percentile = {}".format(np.percentile(CIs, 25)))
    print("50th percentile = {}".format(np.percentile(CIs, 50)))
    print("75th percentile = {}".format(np.percentile(CIs, 75)))
    
    
    # Save results
    print("\nSaving final results.")
    with open(RESULTPATH + description + 'testing_Ci.txt','wb') as f:
        np.savetxt(f, CIs, fmt='%s', delimiter='\t')

#%%===============================================================
#%%===============================================================

if __name__ == '__main__':
    
    # Params
    #=================================================================
    
    # paths ----------------------------------------------------------
    
    #projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
    projectPath = "/home/mtageld/Desktop/KNN_Survival/"
    RESULTPATH_BASE = projectPath + "Results/3_26Sep2017/Gene/"
    
    # dataset and description
    sites = ["GBMLGG", "BRCA", "KIPAN", "MM"]
    dtypes = ["Gene", ] #"Integ"]
    
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
            {'KAPPA': 1.0,
            'OPTIM': 'GD',
            'LEARN_RATE': 0.01,
            }
    
    nca_train_params = \
            {'PLOT_STEP': 200,
            'MODEL_SAVE_STEP': 200,
            }
    
    n_feats_kcv_params = \
            {'kcv': 4,
             'shuffles': 2,
             'n_feats_max': 150,
             'norm': norm,
             }
    
    bagging_params = \
            {'min_n_feats': 20,
             'n_subspaces': 100,
             'norm': norm,
             }
    
    elastic_net_params = \
            {'K': 50,
             'VALID_RATIO': 0.5,
             }
    
    # Now run experiment
    #=================================================================
    
    for USE_NCA in [True, False]:
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
                        nca_train_params['BATCH_SIZE'] = 30
                        nca_train_params['MAX_ITIR'] = 5
                    else:
                        nca_train_params['BATCH_SIZE'] = 400
                        nca_train_params['MAX_ITIR'] = 25
                    
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
                                    RESULTPATH=RESULTPATH + description + '/',
                                    k_tune_params=k_tune_params,
                                    knn_params=knn_params,
                                    USE_NCA=USE_NCA,
                                    graphParams=graphParams,
                                    nca_train_params=nca_train_params,
                                    n_feats_kcv_params=n_feats_kcv_params,
                                    bagging_params=bagging_params,
                                    elastic_net_params=elastic_net_params)
                    
