#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:38:22 2017

@author: mtageld
"""

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
#sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

import _pickle
from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA
from bayes_opt import BayesianOptimization as bayesopt

import NCA_model_10Oct2017 as nca
import KNNSurvival as knn
from pandas import DataFrame as df

#%% ===========================================================================
# Combining results
#==============================================================================

def combine_results(resultpath_base, n_folds=30):
    """
    combine accuracy results from various experiments
    """
    
    print("\nCombining results ...")
    
    idxNames = ['method', 'PCA', 'NCA', 'site', 'dtype']
    folds = list(np.arange(n_folds))
    folds = ['fold_' + str(j) for j in folds]
    idxNames.extend(folds)
    idxNames.extend(['median', 'mean', '25%ile', '75%ile', 'stdev'])
    
    CIs = df(index=idxNames)
    
    print("\nsite\tdtype\tmethod\tPCA\tNCA")
    print("-------------------------------")
    
    for site in ["GBMLGG", "BRCA", "KIPAN", "MM"]:
        for dtype in ["Integ", "Gene"]:
            for method in ["cumulative-time", "non-cumulative"]:
                for USE_PCA in ["False", "True"]:
                    for USE_NCA in ["False", "True"]:
                        
                        try:
                            
                            resultpath = resultpath_base + method + "_" + USE_NCA + "NCA_" + USE_PCA + "PCA/" + \
                                         site + "_" + dtype + "_/" + site + "_" + dtype + "_testing_Ci.txt"
                            
                            ci = np.loadtxt(resultpath, delimiter='\t').reshape([n_folds,])
                            ci_merge = [method, USE_PCA, USE_NCA, site, dtype]
                            ci_merge.extend(ci)
                            ci_merge.extend([np.median(ci), np.mean(ci), \
                                             np.percentile(ci, 25), np.percentile(ci, 75), \
                                             np.std(ci)])
                            
                            
                            CIs[CIs.shape[1]] = ci_merge
                            
                            print("{}\t{}\t{}\t{}\t{}".format(site, dtype, method, USE_PCA, USE_NCA))
                        
                        except FileNotFoundError:
                            pass
    # now save
    CIs.to_csv(resultpath_base + "results_merged.tab", sep="\t")

    print("\nDONE!")

#%% ===========================================================================
# Primary method - get cross validation accuracy
#==============================================================================

def get_cv_accuracy(dpath, site, dtype, description,
                    RESULTPATH,
                    k_tune_params={},
                    USE_NCA=False,
                    graphParams={},
                    nca_train_params={},
                    USE_PCA=False,
                    USE_BAGGING=False,
                    bagging_params={},
                    bo_lims={},
                    bo_expl={},
                    bo_params={}):
    
    """
    Get KNN cross validation accuracy with or without PCA and NCA
    """
    
    # Get a dict of function params and save
    params_all = locals()
    with open(RESULTPATH + description + \
                      'params_all.pkl','wb') as f:
        _pickle.dump(params_all, f)
    
    # result directory structure
    RESULTPATH_NCA = RESULTPATH + "nca/"
    RESULTPATH_KNN = RESULTPATH + "knn/"
    LOADPATH = None
    os.system('mkdir ' + RESULTPATH_NCA)
    os.system('mkdir ' + RESULTPATH_KNN)
    
    # Instantiate a KNN survival model.
    knnmodel = knn.SurvivalKNN(RESULTPATH_KNN, description=description) 
    
    # instantiate NCA model    
    if USE_NCA:
        ncamodel = nca.SurvivalNCA(RESULTPATH_NCA, 
                                   description = description, 
                                   LOADPATH = LOADPATH)
    
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
                                                        norm = k_tune_params['norm'])
            # Predict validation set
            _, Ci = knnmodel.predict(neighbor_idxs,
                                     Survival_train=T_train, 
                                     Censored_train=C_train, 
                                     Survival_test =T_valid, 
                                     Censored_test =C_valid, 
                                     K=k_tune_params['K_init'], 
                                     Method = k_tune_params['Method'])
            
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
    Data = loadmat(dpath)
    Features = Data[dtype + '_X'].copy()
    N = Features.shape[0]
    Survival = Data['Survival'].reshape([N,])
    Censored = Data['Censored'].reshape([N,])
    fnames = Data[dtype + '_Symbs']
    Data = None
    
    if USE_NCA:
        # build computational graph for NCA model
        graphParams['dim_input'] = Features.shape[1]
        ncamodel.build_computational_graph(COMPUT_GRAPH_PARAMS=graphParams)

    
    with open(dpath.split('.mat')[0] + '_splitIdxs.pkl','rb') as f:
        splitIdxs = _pickle.load(f)
    
    # Go through folds, optimize and get accuracy
    #==========================================================================
    
    # initialize
    n_folds = len(splitIdxs['train'])
    CIs = np.zeros([n_folds])
    
    #
    # itirate through folds
    #
    
    #fold = 0
    for fold in range(n_folds):
            
        print("\nfold {} of {}\n".format(fold, n_folds-1))
        
        # Isolate various sets
        # Note, this is done for each loop
        # since they will be modified locally in each outer loop
        X = Features.copy()
        x_train = X[splitIdxs['train'][fold], :]
        x_valid = X[splitIdxs['valid'][fold], :]
        x_test = X[splitIdxs['test'][fold], :]
        X = None
        
        #%% ===================================================================
        # Unsupervised dimensionality reduction - PCA
        #======================================================================
        
        if USE_PCA:
            
            print("\nFinding optimal number of PC's.")  
            
            # Find optimal number of PC's           
            pca = PCA()
            x_train = pca.fit_transform(x_train)
            x_valid = pca.transform(x_valid)
            x_test = pca.transform(x_test)
            
            # keep optimal number of PC's
            numpc_optim = _get_numpc_optim(feats_train=x_train,
                                           feats_valid=x_valid,
                                           T_train=Survival[splitIdxs['train'][fold]],
                                           C_train=Censored[splitIdxs['train'][fold]],
                                           T_valid=Survival[splitIdxs['valid'][fold]],
                                           C_valid=Censored[splitIdxs['valid'][fold]])
            x_train = x_train[:, 0:numpc_optim]
            x_valid = x_valid[:, 0:numpc_optim]
            x_test = x_test[:, 0:numpc_optim]
        
        #%% ===================================================================
        # Supervized dimensionality reduction - NCA
        #======================================================================
        
        if USE_NCA:
            
            #%% ---------------------------------------------------------------
            # Bayesian optimization of NCA hyperparameters
            #------------------------------------------------------------------
            
            print("\nBayesian Optimization of NCA hyperparameters.\n")            
            
            nca_train_params['MONITOR'] = True #False
            
            def run_nca(ALPHA, LAMBDA, SIGMA, DROPOUT_FRACTION):
                
                """
                Wrapper to run NCA and fetch validation accuracy using
                specified tunable hyperparameters                
                """
                
                graph_hyperparams = {'ALPHA': ALPHA,
                                     'LAMBDA': LAMBDA,
                                     'SIGMA': SIGMA,
                                     'DROPOUT_FRACTION': DROPOUT_FRACTION,
                                     }
                      
#                W = ncamodel.train(features = x_train,
#                                   survival = Survival[splitIdxs['train'][fold]],
#                                   censored = Censored[splitIdxs['train'][fold]],
#                                   features_valid = x_valid,
#                                   survival_valid = Survival[splitIdxs['valid'][fold]],
#                                   censored_valid = Censored[splitIdxs['valid'][fold]],
#                                   graph_hyperparams = graph_hyperparams,
#                                   **nca_train_params)
#                
                ncamodel.reset_TrainHistory()
#                
#                # transform
#                x_train_transformed = np.dot(x_train, W)
#                x_valid_transformed = np.dot(x_valid, W)
#                
#                # get neighbor indices    
#                neighbor_idxs = knnmodel._get_neighbor_idxs(x_valid_transformed, 
#                                                            x_train_transformed, 
#                                                            norm = nca_train_params['norm'])
#                
#                # Predict validation set
#                _, Ci = knnmodel.predict(neighbor_idxs,
#                                         Survival_train=Survival[splitIdxs['train'][fold]], 
#                                         Censored_train=Censored[splitIdxs['train'][fold]], 
#                                         Survival_test = Survival[splitIdxs['valid'][fold]], 
#                                         Censored_test = Censored[splitIdxs['valid'][fold]], 
#                                         K = nca_train_params['K'], 
#                                         Method = nca_train_params['Method'])
                      
                _, Ci_valid = ncamodel.train(features = x_train,
                                             survival = Survival[splitIdxs['train'][fold]],
                                             censored = Censored[splitIdxs['train'][fold]],
                                             features_valid = x_valid,
                                             survival_valid = Survival[splitIdxs['valid'][fold]],
                                             censored_valid = Censored[splitIdxs['valid'][fold]],
                                             graph_hyperparams = graph_hyperparams,
                                             **nca_train_params)

                return Ci_valid
            
            #
            # Run core bayesopt model
            #
            
            bo = bayesopt(run_nca, bo_lims)
            bo.explore(bo_expl)
            bo.maximize(init_points = bo_params['init_points'], 
                        n_iter = bo_params['n_itir'])
                        
            # fetch optimal params
            Optim_params = bo.res['max']['max_params']
            ALPHA_OPTIM = Optim_params['ALPHA']
            LAMBDA_OPTIM = Optim_params['LAMBDA']
            SIGMA_OPTIM = Optim_params['SIGMA']
            DROPOUT_FRACTION_OPTIM = Optim_params['DROPOUT_FRACTION']

            print("\tOptimal NCA params:")
            print("\t--------------------")
            print("\tALPHA\tLAMBDA\tSIGMA\tDROPOUT_FRACTION")
            print("\t{}\t{}\t{}\t".format(\
                ALPHA_OPTIM, LAMBDA_OPTIM, SIGMA_OPTIM, 
                DROPOUT_FRACTION_OPTIM))

            #%%----------------------------------------------------------------
            # Learn final NCA matrix
            #------------------------------------------------------------------
            
            print("\nLearning final NCA matrix\n")
            
            nca_train_params['MONITOR'] = True
            
            graph_hyperparams = {'ALPHA': ALPHA_OPTIM,
                                 'LAMBDA': LAMBDA_OPTIM,
                                 'SIGMA': SIGMA_OPTIM,
                                 'DROPOUT_FRACTION': DROPOUT_FRACTION_OPTIM,
                                 }
    
            # Learn NCA matrix
            W = ncamodel.train(features = x_train,
                               survival = Survival[splitIdxs['train'][fold]],
                               censored = Censored[splitIdxs['train'][fold]],
                               features_valid = x_valid,
                               survival_valid = Survival[splitIdxs['valid'][fold]],
                               censored_valid = Censored[splitIdxs['valid'][fold]],
                               graph_hyperparams = graph_hyperparams,
                               **nca_train_params)    
                               
            # Ranks features
            if not USE_PCA:
                ncamodel.rankFeats(W, fnames, rank_type="weights", 
                                   PLOT=nca_train_params['PLOT'])
            
            # Transform features according to learned nca model
            x_train = np.dot(x_train, W)
            x_valid = np.dot(x_valid, W)
            x_test = np.dot(x_test, W)
            
        #%% ===================================================================    
        # Tune K
        #======================================================================
                                         
        # Get neighbor indices    
        neighbor_idxs = knnmodel._get_neighbor_idxs(\
                            x_valid, x_train, 
                            norm = k_tune_params['norm'])
    
        print("\tK \t Ci")
        
        CIs_k = np.zeros([len(k_tune_params['Ks'])])
        for kidx, K in enumerate(k_tune_params['Ks']):
        
            # Predict validation set
            _, Ci = knnmodel.predict(\
                                 neighbor_idxs=neighbor_idxs,
                                 Survival_train=Survival[splitIdxs['train'][fold]],
                                 Censored_train=Censored[splitIdxs['train'][fold]],
                                 Survival_test=Survival[splitIdxs['valid'][fold]],
                                 Censored_test=Censored[splitIdxs['valid'][fold]],
                                 K=K,
                                 Method=k_tune_params['Method'])
        
            CIs_k[kidx] = Ci
        
            print("\t{} \t {}".format(K, round(Ci, 3)))
            
        K_optim = k_tune_params['Ks'][np.argmax(CIs_k)]
        print("\nK_optim = {}".format(K_optim))
           
        #%% ===================================================================    
        # Get final accuracy
        #======================================================================
        
        print("\nGetting accuracy.") 
        
        # combined training and validation sets
        combinedIdxs = splitIdxs['train'][fold] + splitIdxs['valid'][fold]
        
        if USE_BAGGING:
            _, ci = knnmodel.predict_with_bagging(\
                        X_test=x_test,
                        X_train=np.concatenate((x_train, x_valid), axis=0),
                        Survival_train=Survival[combinedIdxs],
                        Censored_train=Censored[combinedIdxs],
                        Survival_test=Survival[splitIdxs['test'][fold]],
                        Censored_test=Censored[splitIdxs['test'][fold]],
                        **bagging_params,
                        K=K_optim,
                        Method=k_tune_params['Method'],
                        norm=k_tune_params['norm'])  
        else:
            neighbor_idxs = knnmodel._get_neighbor_idxs(\
                                x_test, np.concatenate((x_train, x_valid), axis=0), 
                                norm = k_tune_params['norm'])            
            _, ci = knnmodel.predict(\
                                     neighbor_idxs=neighbor_idxs,
                                     Survival_train=Survival[combinedIdxs],
                                     Censored_train=Censored[combinedIdxs],
                                     Survival_test=Survival[splitIdxs['test'][fold]],
                                     Censored_test=Censored[splitIdxs['test'][fold]],
                                     K=K_optim,
                                     Method=k_tune_params['Method'])
        
        # record result
        CIs[fold] = ci        
        print("Ci = {}".format(round(ci, 3)))
    
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
    
    projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
    #projectPath = "/home/mtageld/Desktop/KNN_Survival/"
    RESULTPATH_BASE = projectPath + "Results/11_11Oct2017/"
    
    # dataset and description
    sites = ["GBMLGG", "BRCA", "KIPAN", "MM"]
    dtypes = ["Integ", ] #"Gene"]
    
    K_init = 35
    norm = 2
    Methods = ['cumulative-time', 'non-cumulative']
    
    # KNN params ----------------------------------------------------
    
    k_tune_params = {'K_init': K_init,
                     'Ks': list(np.arange(10, 160, 10)),
                     'norm': norm,
                     }
                     
    USE_BAGGING = True
    
    bagging_params = {'n_bags': 50,
                      'feats_per_bag': None
                      }
    
    # NCA params  ---------------------------------------------------
    
    graphParams = \
            {'OPTIM': 'Adam',
            'LEARN_RATE': 0.002, #0.002,
            'per_split_feats': 500,
            'transform': 'linear', #'ffnn', 
            'DEPTH': 2,
            'MAXWIDTH': 300,
            'dim_output': 2,
            'NONLIN': 'ReLU',
            'ROTATE': False,
            }
    
    nca_train_params = \
            {'PLOT_STEP': 200,
            'MODEL_SAVE_STEP': 200,
            'BATCH_SIZE': 400,
            'MAX_ITIR': 25,
            'MODEL_BUFFER': 4,
            'EARLY_STOPPING': True,
            'PLOT': True,
            'K': K_init,
            'norm': norm,
            }
             
    # Bayesopt params  ---------------------------------------------------
             
    # limits of interval to explore
    bo_lims = {'ALPHA': (0, 1),
               'LAMBDA': (0, 1),
               'SIGMA': (0.2, 15),
               'DROPOUT_FRACTION': (0, 0.7),
               }
    
    # initial points to explore
    bo_expl = {'ALPHA': [0, 0, 1, 0, 0],
               'LAMBDA': [0, 1, 0, 0, 0],
               'SIGMA': [1, 1, 1, 5, 0.5],
               'DROPOUT_FRACTION': [0, 0.5, 0, 0.5, 0],
               }
    
    # other bayesopt params
    bo_params = {'init_points': 2,
                 'n_itir': 15,
                 }
        
    # Now run experiment
    #=================================================================
    
    for USE_NCA in [True, False]:
        for USE_PCA in [False, ]: #True]:
            for Method in Methods:
            
                # pass params to dicts
                k_tune_params['Method'] = Method
                nca_train_params['Method'] = Method
                
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
                        
                        #raise Exception("on purpose.")
                        
                        # get cv accuracy
                        get_cv_accuracy(dpath=dpath, site=site, dtype=dtype,
                                        description=description,
                                        RESULTPATH=RESULTPATH + description + '/',
                                        k_tune_params=k_tune_params,
                                        USE_NCA=USE_NCA,
                                        graphParams=graphParams,
                                        nca_train_params=nca_train_params,
                                        USE_PCA=USE_PCA,
                                        USE_BAGGING=USE_BAGGING,
                                        bagging_params=bagging_params,
                                        bo_lims=bo_lims,
                                        bo_expl=bo_expl,
                                        bo_params=bo_params)
                                            

    # Combine results from all experiments
    #=================================================================
    
    combine_results(RESULTPATH_BASE)

#%%
#%%
#%%

