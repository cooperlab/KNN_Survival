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


# Params
#=================================================================

# paths ----------------------------------------------------------

projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
#projectPath = "/home/mtageld/Desktop/KNN_Survival/"
RESULTPATH_BASE = projectPath + "Results/tmp/"

# dataset and description
sites = ["GBMLGG", ] #"BRCA", "KIPAN",] # "LUSC"]
dtypes = ["Integ",] # "Gene]

norm = 2
Methods = ['cumulative-time',] # 'non-cumulative']

# KNN params ----------------------------------------------------

k_tune_params = {'kcv': 4,
                 'shuffles': 5,
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
        'OPTIM': 'GD',
        'LEARN_RATE': 0.01
        }

nca_train_params = \
        {'BATCH_SIZE': 200, #40,
        'PLOT_STEP': 200,
        'MODEL_SAVE_STEP': 200,
        'MAX_ITIR': 2,
        }

n_feats_kcv_params = \
        {'kcv': 4,
         'shuffles': 2,
         'n_feats_max': 100,
         'K': 30,
         'norm': norm,
         }

# Now run experiment
#=================================================================


USE_NCA = True
Method = Methods[0]

# pass params to dicts
k_tune_params['Method'] = Method
knn_params['Method'] = Method
n_feats_kcv_params['Method'] = Method

# Itirate through datasets

RESULTPATH = RESULTPATH_BASE + \
             Method + "_" + \
             str(USE_NCA) + "NCA/"
os.system("mkdir " + RESULTPATH)

dtype = dtypes[0]

for site in sites:
    
    description = site +"_"+ dtype +"_"
    dpath = projectPath + "Data/SingleCancerDatasets/"+ site+"/"+ \
            site +"_"+ dtype+"_Preprocessed.mat"
    
    # create output directory
    os.system('mkdir ' + RESULTPATH + description)
    
    
    #==========================================================================
    # ***  M A I N  M E T H O D  ***
    #==========================================================================
    
    
    """
    Get KNN cross validation accuracy with or without ensembles and NCA
    """
    
    ## Get a dict of function params and save
    #params_all = locals()
    #with open(RESULTPATH + description + \
    #                  'params_all.pkl','wb') as f:
    #    _pickle.dump(params_all, f)
    
    
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
    
    # -------------------------------------------
    # *** I N P U T S ***
    # -------------------------------------------

    import SurvivalUtils as sUtils

    n_subspaces = 10

    if Method == "cumulative-hazard":
        prediction_type = "risk"
    else:
        prediction_type = "survival_time"

    # -------------------------------------------
    
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
        
        # Get optimal no of features to use
        #===================================
        
        # get accuracy
        print("\nOptimizing no of features to use.")
        
        # sort X by absolute feature weights
        sort_idxs = np.flip(np.abs(w).argsort(), axis=0)
        X = X[:, sort_idxs]
        
        # Get optimal no of feats
        _, n_feats_optim = \
            knnmodel.get_optimal_n_feats(\
                X=X[optimIdxs, :], 
                T=Survival[optimIdxs], 
                C=Censored[optimIdxs],
                **n_feats_kcv_params)

        # Find optimal K to use
        #====================================

        print("\nFinding optimal K to use.")

        _, K_optim = knnmodel.tune_k(X[optimIdxs, :],
                Survival[optimIdxs],
                Censored[optimIdxs],
                **k_tune_params)

        
        print("outer_fold \t fold \t Ci")

        for fold in range(n_folds):
        
            # Isolate patients belonging to fold
        
            idxs_train = splitIdxs['fold_cv_train'][outer_fold][fold]
            idxs_test = splitIdxs['fold_cv_test'][outer_fold][fold]
            
            X_test = X[idxs_test, :]
            X_train = X[idxs_train, :]
            Survival_train = Survival[idxs_train]
            Censored_train = Censored[idxs_train]
            Survival_test = Survival[idxs_test]
            Censored_test = Censored[idxs_test]
        
    
            # Get accuracy using bagged subspaces KNN approach
            #==================================================
    
            # initialize
            preds = np.zeros([X_test.shape[0], n_subspaces)

            for subspace in range(n_subspaces):

                print('\t\tSubspace {} of {}'.format(subspace, n_subspaces-1))
                
                # maximum feature index
                fidx_max = np.random.randint(10, n_feats_optim+1)
    
                # Get neighbor indices    
                neighbor_idxs = knnmodel._get_neighbor_idxs(\
                        X_test[:, 0:fidx_max], 
                        X_train[:, 0:fidx_max], 
                        norm = norm)
            
                # Predict testing set
                t_test, _ = self.predict(neighbor_idxs,
                                     Survival_train, Censored_train, 
                                     K = K_optim, Method = Method)
               
                preds[:, subspace] = t_test

            # Aggregate prediction and get ci
            t_test = np.mean(preds, axis=1)
            CIs[fold, outer_fold] = \
                    sUtils.c_index(t_test, Survival_test, Censored_test, 
                                   prediction_type= prediction_type)

        
        #====================================================================
        
        #ci, _ = knnmodel.cv_accuracy(X[:, 0:n_feats_optim], 
        #                             Survival, Censored, \
        #                             splitIdxs, outer_fold=outer_fold,\
        #                             tune_params=k_tune_params)
        #
        #CIs[:, outer_fold] = ci
           
        
    print("\nAccuracy")
    print("------------------------")
    print("25th percentile = {}".format(np.percentile(CIs, 25)))
    print("50th percentile = {}".format(np.percentile(CIs, 50)))
    print("75th percentile = {}".format(np.percentile(CIs, 75)))
    
    
    # Save results
    print("\nSaving final results.")
    with open(RESULTPATH + description + 'testing_Ci.txt','wb') as f:
        np.savetxt(f, CIs, fmt='%s', delimiter='\t')
