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
from sklearn.decomposition import PCA

import NCA_model as nca
import KNNSurvival as knn

#%% ===========================================================================
# Primary method - get cross validation accuracy
#==============================================================================

def get_cv_accuracy(dpath, site, dtype, description,
                    RESULTPATH,
                    k_tune_params={},
                    knn_params={},
                    USE_NCA=False,
                    graphParams={},
                    nca_train_params={},
                    elastic_net_params={},
                    USE_PCA=False,
                    USE_BAGGING=False,
                    bagging_params={}):
    
    """
    Get KNN cross validation accuracy with or without PCA and NCA
    """
    
    # Get a dict of function params and save
    params_all = locals()
    with open(RESULTPATH + description + \
                      'params_all.pkl','wb') as f:
        _pickle.dump(params_all, f)
    
    #%% =======================================================================
    # Define relevant methods
    #==========================================================================  
    
    def _get_numpc_optim(feats_train, feats_valid,
                         T_train, C_train,
                         T_valid, C_valid):
        
        """
        Given PCA-transformed traing and validation sets,
        find the optimal no of principal components to 
        maximize the Ci
        """
        print("\nFinding optimal number of PC's.")   
        print("\n\tnumpc\tCi")
        print("\t--------------")
        
        cis = []
        
        numpc_max = np.min([feats_train.shape[1], 200])
        
        for numpc in range(4, numpc_max, 4):
            feats_train_new = feats_train[:, 0:numpc]
            feats_valid_new = feats_valid[:, 0:numpc]
            # get neighbor indices    
            neighbor_idxs = knnmodel._get_neighbor_idxs(feats_valid_new, 
                                                        feats_train_new, 
                                                        norm = norm)
            # Predict validation set
            _, Ci = knnmodel.predict(neighbor_idxs,
                                     Survival_train=T_train, 
                                     Censored_train=C_train, 
                                     Survival_test =T_valid, 
                                     Censored_test =C_valid, 
                                     K=elastic_net_params['K'], 
                                     Method = Method)
            
            cis.append([numpc, Ci])
            print("\t{}\t{}".format(numpc, Ci))
        
        # now get optimal no of PC's
        cis = np.array(cis)
        numpc_optim = cis[cis[:,1].argmax(), 0]
        print("\nnumpc_optim = {}".format(round(numpc_optim, 3)))
            
        return int(numpc_optim)

    #%% =======================================================================
    # Begin main body
    #==========================================================================    
    
    print("\n--------------------------------------")
    print("Getting cv accuracy: {}, {}".format(site, dtype))
    print("--------------------------------------\n")
    
    print("Loading data.")
    
    #Data = loadmat(dpath)
    #Features = Data[dtype + '_X'] 
    #N = Features.shape[0]

    
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
    #==========================================================================
    
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
        
        # Note, this is done for each outer loop
        # since they will be modified locally in each outer loop
        print("Loading data ...")
        Data = loadmat(dpath)
        X = Data[dtype + '_X'].copy()
        N = X.shape[0]
        Survival = Data['Survival'].reshape([N,])
        Censored = Data['Censored'].reshape([N,])
        Data = None
        
        # Isolate optimization set (and divide into training and validation)
        optimIdxs = splitIdxs['idx_optim'][outer_fold]
        
        if (USE_NCA or USE_PCA):
            stoppoint = int(elastic_net_params['VALID_RATIO'] * len(optimIdxs))
            optimIdxs_valid = optimIdxs[0:stoppoint]
            optimIdxs_train = optimIdxs[stoppoint:]
            x_train = X[optimIdxs_train, :]
            x_valid = X[optimIdxs_valid, :]
        
        #%% ===================================================================
        # Unsupervised dimensionality reduction - PCA
        #======================================================================
        
        if USE_PCA:
            
            # Find optimal number of PC's           
            pca = PCA()
            x_train = pca.fit_transform(x_train)
            x_valid = pca.transform(x_valid)
            
            # keep optimal number of PC's
            numpc_optim = _get_numpc_optim(feats_train=x_train,
                                           feats_valid=x_valid,
                                           T_train=Survival[optimIdxs_train],
                                           C_train=Censored[optimIdxs_train],
                                           T_valid=Survival[optimIdxs_valid],
                                           C_valid=Censored[optimIdxs_valid])
            x_train = x_train[:, 0:numpc_optim]
            x_valid = x_valid[:, 0:numpc_optim]
            
            # Now learn final PC matrix on full optimization set
            print("\nLearning final PCA matrix.")            
            pca = PCA(n_components=numpc_optim)
            pca.fit(X[optimIdxs, :])
            X = pca.transform(X)
            
        
        #%% ===================================================================
        # Supervized dimensionality reduction - NCA
        #======================================================================
        
        if USE_NCA:
            
            # instantiate NCA model
            ncamodel = nca.SurvivalNCA(RESULTPATH_NCA, 
                                       description = description, 
                                       LOADPATH = LOADPATH)
            #                          
            # Finding optimal values for ALPHA and LAMBDA (regularization)
            #
            
            ALPHAS = np.arange(0, 1.1, 0.2)
            LAMBDAS = np.arange(0, 1.1, 0.2)
    
            cis = []
            
            for ALPHA in ALPHAS:
                for LAMBDA in LAMBDAS:
                    
                    if ((LAMBDA == 0) and (ALPHA > ALPHAS.min())):
                        continue
            
                    graphParams['ALPHA'] = ALPHA
                    graphParams['LAMBDA'] = LAMBDA
                    
                    w = ncamodel.train(features = x_train,
                                       survival = Survival[optimIdxs_train],
                                       censored = Censored[optimIdxs_train],
                                       COMPUT_GRAPH_PARAMS = graphParams,
                                       **nca_train_params)
                    W = np.zeros([len(w), len(w)])
                    np.fill_diagonal(W, w)
                    
                    ncamodel.reset_TrainHistory()
                    
                    # transform
                    x_valid_transformed = np.dot(x_valid, W)
                    x_train_transformed = np.dot(x_train, W)
                    
                    # get neighbor indices    
                    neighbor_idxs = knnmodel._get_neighbor_idxs(x_valid_transformed, 
                                                                x_train_transformed, 
                                                                norm = norm)
                    
                    # Predict validation set
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
            
            #           
            # Learn final NCA matrix on optimization set
            #
            
            print("\nLearning final NCA matrix\n")
            
            graphParams['ALPHA'] = ALPHA_OPTIM
            graphParams['LAMBDA'] = LAMBDA_OPTIM
    
            # Learn NCA matrix
            w = ncamodel.train(features = X[optimIdxs, :],
                               survival = Survival[optimIdxs],
                               censored = Censored[optimIdxs],
                               COMPUT_GRAPH_PARAMS = graphParams,
                               **nca_train_params)
            W = np.zeros([len(w), len(w)])
            np.fill_diagonal(W, w)
            
            # Transform features according to learned nca model
            X = np.dot(X, W)
            
        #%% ===================================================================    
        # Now get accuracy
        #======================================================================
        
        print("\nGetting accuracy.") 
        ci, _ = knnmodel.cv_accuracy(X, Survival, Censored, 
                                     splitIdxs, outer_fold=outer_fold,
                                     k_tune_params=k_tune_params,
                                     USE_BAGGING=USE_BAGGING,
                                     bagging_params=bagging_params)
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


#%%============================================================================
#%%============================================================================
#%%============================================================================

if __name__ == '__main__':
    
    # Params
    #=================================================================
    
    # paths ----------------------------------------------------------
    
    #projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
    projectPath = "/home/mtageld/Desktop/KNN_Survival/"
    RESULTPATH_BASE = projectPath + "Results/6_28Sep2017/"
    
    # dataset and description
    sites = ["GBMLGG", "BRCA", "KIPAN", "MM"]
    dtypes = ["Gene", "Integ"]
    
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
    
    USE_BAGGING = True
    
    bagging_params = {'n_bags': 50,
                      'feats_per_bag': None
                      }
    
    # NCA params  ---------------------------------------------------
    
    graphParams = \
            {'OPTIM': 'GD',
            'LEARN_RATE': 0.01,
            'per_split_feats': 500,
            }
    
    nca_train_params = \
            {'PLOT_STEP': 200,
            'MODEL_SAVE_STEP': 200,
            'BATCH_SIZE': 400,
            'MAX_ITIR': 25,
            }
    
    elastic_net_params = \
            {'K': 50,
             'VALID_RATIO': 0.5,
             }
        
    # Now run experiment
    #=================================================================
    
    for USE_NCA in [True, False]:
        for Method in Methods:
            for USE_PCA in [False, True]:
            
                # pass params to dicts
                k_tune_params['Method'] = Method
                knn_params['Method'] = Method
                
                # Itirate through datasets
                
                RESULTPATH = RESULTPATH_BASE + \
                             Method + "_" + \
                             str(USE_NCA) + "NCA_" + \
                             str(USE_PCA) + "PCA/"
                success = os.system("mkdir " + RESULTPATH)
                
                if success != 0:
                    print("Folder exists, experiment already done.")
                    continue
                
                for dtype in dtypes:
                    for site in sites:
                        
                        if (site == "MM") and (dtype == "Integ"):
                            continue
                            
                        if (USE_PCA and (not USE_NCA)):
                            continue
                        
                        if ((not USE_PCA) and (not USE_NCA)):
                            continue

                        if ((dtype == "Gene") and (not USE_PCA)):
                            continue
                        
                        #if (dtype == "Gene") and (not USE_PCA):
                        #    nca_train_params['BATCH_SIZE'] = 100
                        #    nca_train_params['MAX_ITIR'] = 8
                        #else:
                        #    nca_train_params['BATCH_SIZE'] = 400
                        #    nca_train_params['MAX_ITIR'] = 25

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
                                        elastic_net_params=elastic_net_params,
                                        USE_PCA=USE_PCA,
                                        USE_BAGGING=USE_BAGGING,
                                        bagging_params=bagging_params)
                                            
    
#%%
#%%
#%%

