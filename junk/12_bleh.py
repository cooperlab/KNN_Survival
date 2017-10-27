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
#from sklearn.decomposition import PCA
#from bayes_opt import BayesianOptimization as bayesopt

import NCA_model_experimental as nca
import KNNSurvival as knn
#from pandas import DataFrame as df

#%%============================================================================
# Params
#==============================================================================

# paths ----------------------------------------------------------

projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
#projectPath = "/home/mtageld/Desktop/KNN_Survival/"
RESULTPATH_BASE = projectPath + "Results/tmp/"

# dataset and description
sites = ["GBMLGG", ] #, "MM"]
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
        'LEARN_RATE': 0.002,
        'per_split_feats': 500,
        'dim_output': 1000,
        'transform': 'linear', #'ffnn', 
        'ROTATE': False,
        'DEPTH': 2,
        'MAXWIDTH': 300,
        'NONLIN': 'ReLU',
        }

nca_train_params = \
        {'PLOT_STEP': 200,
        'MODEL_SAVE_STEP': 200,
        'BATCH_SIZE': 400,
        'MAX_ITIR': 100, # 100,
        'MODEL_BUFFER': 8,
        'EARLY_STOPPING': False, #True,
        'PLOT': True, #True,
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
bo_expl = {'ALPHA': [0, ],
           'LAMBDA': [0, ],
           'SIGMA': [1, ],
           'DROPOUT_FRACTION': [0, ],
           }

# other bayesopt params
bo_params = {'init_points': 1,
             'n_itir': 1,
             }

USE_NCA = True
# for USE_NCA in [True, False]

Method = Methods[0]
# for Method in Methods:
        
# pass params to dicts
k_tune_params['Method'] = Method
nca_train_params['Method'] = Method

# Itirate through datasets
RESULTPATH = RESULTPATH_BASE + \
             Method + "_" + \
             str(USE_NCA) + "NCA/"
success = os.system("mkdir " + RESULTPATH)

if success != 0:
    print("Folder exists, experiment already done.")
#    continue

dtype = dtypes[0]
site = sites[0]
#for dtype in dtypes:
#    for site in sites:
        
description = site +"_"+ dtype +"_"
dpath = projectPath + "Data/SingleCancerDatasets/"+ site+"/"+ \
        site +"_"+ dtype+"_Preprocessed.mat"

# create output directory
os.system('mkdir ' + RESULTPATH + description)

#%% ===========================================================================
# Primary method - get cross validation accuracy
#==============================================================================

print("\n--------------------------------------")
print("Getting cv accuracy: {}, {}".format(site, dtype))
print("--------------------------------------\n")
    
# result directory structure
RESULTPATH_NCA = RESULTPATH + "nca/"
RESULTPATH_KNN = RESULTPATH + "knn/"
LOADPATH = None
os.system('mkdir ' + RESULTPATH_NCA)
os.system('mkdir ' + RESULTPATH_KNN)

# Instantiate a KNN survival model.
knnmodel = knn.SurvivalKNN(RESULTPATH_KNN, description=description) 

# instantiate NCA model    
ncamodel = nca.SurvivalNCA(RESULTPATH_NCA, 
                           description = description, 
                           LOADPATH = LOADPATH)
    

#%% =======================================================================
# Begin main body
#==========================================================================    

print("Loading data.")
Data = loadmat(dpath)
Features = Data[dtype + '_X'].copy()
N = Features.shape[0]
Survival = Data['Survival'].reshape([N,])
Censored = Data['Censored'].reshape([N,])
fnames = Data[dtype + '_Symbs']
Data = None
    
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
    
    #%% ===================================================================
    # Isolate various sets
    #======================================================================
    
    # Note, this is done for each loop
    # since they will be modified locally in each outer loop
    X = Features.copy()
    x_train = X[splitIdxs['train'][fold], :]
    x_valid = X[splitIdxs['valid'][fold], :]
    x_test = X[splitIdxs['test'][fold], :]
    X = None
    
    #%% ===================================================================
    # build computational graph for NCA model
    #======================================================================
    
    graphParams['dim_input'] = Features.shape[1]
    
    epsilon = 1e-3
    w_init = (x_train.max(axis=0) - x_train.min(axis=0))
    w_init = 1 - (w_init / w_init.max()) + epsilon
    graphParams['w_init'] = w_init
    
    ncamodel.build_computational_graph(COMPUT_GRAPH_PARAMS=graphParams)
        
        
    #%% ===================================================================
    # optimization of NCA hyperparameters
    #======================================================================
    
    print("\n Optimization of NCA hyperparameters.\n")            
    
    nca_train_params['MONITOR'] = True #False
    
    def run_nca(ALPHA= 0, LAMBDA= 0.04, SIGMA= 1, DROPOUT_FRACTION= 0):
                
        """
        Wrapper to run NCA and fetch validation accuracy using
        specified tunable hyperparameters                
        """
        
        graph_hyperparams = {'ALPHA': ALPHA,
                             'LAMBDA': LAMBDA,
                             'SIGMA': SIGMA,
                             'DROPOUT_FRACTION': DROPOUT_FRACTION,
                             }
                             
        if graphParams['transform'] == 'linear':

            # Fetch weights (this allows early stopping)                    
            
            W = ncamodel.train(features = x_train,
                               survival = Survival[splitIdxs['train'][fold]],
                               censored = Censored[splitIdxs['train'][fold]],
                               features_valid = x_valid,
                               survival_valid = Survival[splitIdxs['valid'][fold]],
                               censored_valid = Censored[splitIdxs['valid'][fold]],
                               graph_hyperparams = graph_hyperparams,
                               **nca_train_params)
            
            ncamodel.reset_TrainHistory()
            
            # transform
            x_train_transformed = np.dot(x_train, W)
            x_valid_transformed = np.dot(x_valid, W)
            
            # get neighbor indices    
            neighbor_idxs = knnmodel._get_neighbor_idxs(x_valid_transformed, 
                                                        x_train_transformed, 
                                                        norm = nca_train_params['norm'])
            
            # Predict validation set
            _, Ci_valid = knnmodel.predict(neighbor_idxs,
                                     Survival_train=Survival[splitIdxs['train'][fold]], 
                                     Censored_train=Censored[splitIdxs['train'][fold]], 
                                     Survival_test = Survival[splitIdxs['valid'][fold]], 
                                     Censored_test = Censored[splitIdxs['valid'][fold]], 
                                     K = nca_train_params['K'], 
                                     Method = nca_train_params['Method'])
                                     
        else:
            
            # Fetch Ci directly
            ncamodel.reset_TrainHistory()      
            _, Ci_valid = ncamodel.train(features = x_train,
                                         survival = Survival[splitIdxs['train'][fold]],
                                         censored = Censored[splitIdxs['train'][fold]],
                                         features_valid = x_valid,
                                         survival_valid = Survival[splitIdxs['valid'][fold]],
                                         censored_valid = Censored[splitIdxs['valid'][fold]],
                                         graph_hyperparams = graph_hyperparams,
                                         **nca_train_params)

        return Ci_valid
            
    #            #
    #            # Run core bayesopt model
    #            #
    #            
    #            bo = bayesopt(run_nca, bo_lims)
    #            bo.explore(bo_expl)
    #            bo.maximize(init_points = bo_params['init_points'], 
    #                        n_iter = bo_params['n_itir'])
    #                        
    #            # fetch optimal params
    #            Optim_params = bo.res['max']['max_params']
    #            ALPHA_OPTIM = Optim_params['ALPHA']
    #            LAMBDA_OPTIM = Optim_params['LAMBDA']
    #            SIGMA_OPTIM = Optim_params['SIGMA']
    #            DROPOUT_FRACTION_OPTIM = Optim_params['DROPOUT_FRACTION']
    #
    #            print("\tOptimal NCA params:")
    #            print("\t--------------------")
    #            print("\tALPHA\tLAMBDA\tSIGMA\tDROPOUT_FRACTION")
    #            print("\t{}\t{}\t{}\t".format(\
    #                ALPHA_OPTIM, LAMBDA_OPTIM, SIGMA_OPTIM, 
    #                DROPOUT_FRACTION_OPTIM))
        
    lambdas = np.arange(7) * 10e-3
    cis_valid = np.zeros(len(lambdas))
    
    for lidx, LAMBDA in enumerate(lambdas):
        cis_valid[lidx] = run_nca(lambdas)
    
    # fetch optimal params
    ALPHA_OPTIM = 0
    LAMBDA_OPTIM = lambdas[np.argmax(cis_valid)]
    SIGMA_OPTIM = 1
    DROPOUT_FRACTION_OPTIM = 0
    
    print("\tOptimal NCA params:")
    print("\t--------------------")
    print("\tLAMBDA\tSIGMA")
    print("\t{}\t{}".format(LAMBDA_OPTIM, SIGMA_OPTIM))
    
    break



#%%============================================================================
#%%============================================================================
#%%============================================================================
