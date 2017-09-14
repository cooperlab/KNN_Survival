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

import DataManagement as dm
import NCA_model as nca
import KNNSurvival as knn

#%%========================================================================
# Prepare inputs
#==========================================================================

print("Loading and preprocessing data.")

# Load data

#projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
projectPath = "/home/mtageld/Desktop/KNN_Survival/"

dpath = projectPath + "Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
#dpath = projectPath + "Data/SingleCancerDatasets/GBMLGG/Brain_Gene.mat"
#dpath = projectPath + "Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat"

Data = loadmat(dpath)

Features = np.float32(Data['Integ_X'])
#Features = np.float32(Data['Gene_X'])

N, D = Features.shape

if np.min(Data['Survival']) < 0:
    Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1

Survival = np.int32(Data['Survival']).reshape([N,])
Censored = np.int32(Data['Censored']).reshape([N,])
fnames = Data['Integ_Symbs']
#fnames = Data['Gene_Symbs']

# remove zero-variance features
fvars = np.std(Features, 0)
keep = fvars > 0
Features = Features[:, keep]
fnames = fnames[keep]

# params
RESULTPATH = projectPath + "Results/tmp/"
RESULTPATH_NCA = RESULTPATH + "nca/"
RESULTPATH_KNN = RESULTPATH + "knn/"
description = "GBMLGG_Integ_"
LOADPATH = None
#LOADPATH = RESULTPATH_NCA + 'model/' + description + 'ModelAttributes.pkl'

# create subdirs
os.system('mkdir ' + RESULTPATH_NCA)
os.system('mkdir ' + RESULTPATH_KNN)


#%%========================================================================
# Get split indices
#==========================================================================

# Get split indices - entire cohort

K_OPTIM = 2
K = 3
SHUFFLES = 10
splitIdxs = dm.get_balanced_SplitIdxs(Censored, \
                                      K = K,\
                                      SHUFFLES = SHUFFLES,\
                                      USE_OPTIM = True,\
                                      K_OPTIM = K_OPTIM)

# Save split indices for replicability
with open(RESULTPATH + description + \
                  'splitIdxs.pkl','wb') as f:
    _pickle.dump(splitIdxs, f)


#%%============================================================================
# Go through outer folds, optimize and get accuracy
#==============================================================================

#
# define params
#

graphParams = {'ALPHA': 0.5,
               'LAMBDA': 0, 
               'OPTIM': 'GD',
               'LEARN_RATE': 0.01}

nca_train_params = {'BATCH_SIZE': 200, \
                    'PLOT_STEP': 100, \
                    'MODEL_SAVE_STEP': 100, \
                    'MAX_ITIR': 100,
                   }


k_tune_params = {'kcv': 5,
                 'shuffles': 1,
                 'Ks': list(np.arange(10, 160, 10)),
                }

#
# initialize
#

CIs_X = np.zeros([K * SHUFFLES, K_OPTIM])
K_optim_X = np.zeros([K_OPTIM])

CIs_XA = np.zeros([K * SHUFFLES, K_OPTIM])
K_optim_XA = np.zeros([K_OPTIM])

#
# itirate through folds
#

for outer_fold in range(K_OPTIM):

    print("\nOuter fold {} of {}\n".format(outer_fold, K_OPTIM-1))

    # Isolate optimization set 
    optimIdxs = splitIdxs['idx_optim'][outer_fold]


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
    Features_transformed = np.dot(Features, W)
   

    # Get model accuracy
    #=====================
    
    # Instantiate a KNN survival model.
    knnmodel = knn.SurvivalKNN(RESULTPATH_KNN, description = description)
    
    
    # get accuracy on non-nca-transformed set
    print("\nGetting accuracy on original X")
    
    ci, k_optim = knnmodel.cv_accuracy(Features, Survival, Censored, \
                                       splitIdxs, outer_fold = outer_fold,\
                                       tune_params = k_tune_params)
    CIs_X[:, outer_fold] = ci
    K_optim_X[outer_fold] = k_optim
    
    
    
    # get accuracy on nca-transformed set
    print("\nGetting accuracy on XA")
    
    ci, k_optim = knnmodel.cv_accuracy(Features_transformed, Survival, Censored, \
                                       splitIdxs, outer_fold = outer_fold,\
                                       tune_params = k_tune_params)
    CIs_XA[:, outer_fold] = ci
    K_optim_XA[outer_fold] = k_optim


print("\nAccuracy on original X")
print("------------------------")
print("25th percentile = {}".format(np.percentile(CIs_X, 25)))
print("50th percentile = {}".format(np.percentile(CIs_X, 50)))
print("75th percentile = {}".format(np.percentile(CIs_X, 75)))


print("\nAccuracy on XA")
print("------------------------")
print("25th percentile = {}".format(np.percentile(CIs_XA, 25)))
print("50th percentile = {}".format(np.percentile(CIs_XA, 50)))
print("75th percentile = {}".format(np.percentile(CIs_XA, 75)))


# Save results
Results = {'graphParams': graphParams,
           'nca_train_params': nca_train_params,
           'k_tune_params': k_tune_params,
           'CIs_K_X': CIs_K_X,
           'K_optim_X': K_optim_X,
           'CIs_X': CIs_X,
           'CIs_K_XA': CIs_K_XA,
           'K_optim_XA': K_optim_XA,
           'CIs_XA': CIs_XA,
           }


print("\nSaving final results.")
with open(RESULTPATH + description + \
                  'Results.pkl','wb') as f:
    _pickle.dump(Results, f)
