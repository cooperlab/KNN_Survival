# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:54:33 2017

@author: mohamed
"""

import os
import sys
#sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

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
RESULTPATH_NCA = projectPath + "Results/tmp/nca/"
RESULTPATH_KNN = projectPath + "Results/tmp/knn/"
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
splitIdxs = dm.get_balanced_SplitIdxs(Censored, OPTIM_RATIO = 0.5,\
                                      K = 3,\
                                      SHUFFLES = 10)

# Isolate optimization set 
optimIdxs = splitIdxs['idx_optim_train'] + splitIdxs['idx_optim_valid']


#%%============================================================================
# Learn NCA matrix on optimization set
#==============================================================================

# instantiate
ncamodel = nca.SurvivalNCA(RESULTPATH_NCA, description = description, \
                           LOADPATH = LOADPATH)

# train
graphParams = {'ALPHA': 0.5,
               'LAMBDA': 0, 
               'OPTIM': 'GD',
               'LEARN_RATE': 0.01}
                           
ncamodel.train(features = Features[optimIdxs, :],
               survival = Survival[optimIdxs],
               censored = Censored[optimIdxs],
               COMPUT_GRAPH_PARAMS = graphParams,
               BATCH_SIZE = 200,
               PLOT_STEP = 100,
               MODEL_SAVE_STEP = 100,
               MAX_ITIR = 100)

# get feature ranks
ncamodel.rankFeats(Features, fnames, rank_type = "weights")
ncamodel.rankFeats(Features, fnames, rank_type = "stdev")


#%%============================================================================
# Transform features according to learned nca model
#==============================================================================

# get learned weights
w = np.load(RESULTPATH_NCA + 'model/' + description + 'featWeights.npy')  
W = np.zeros([len(w), len(w)])
np.fill_diagonal(W, w)

# transform
Features_transformed = np.dot(Features, W)

def get_accuracy(X):

    """Get model accuracy using KNN"""

    #%%============================================================================
    # Tune KNN model on optimization set
    #==============================================================================
    
    # instantiate
    knnmodel = knn.SurvivalKNN(RESULTPATH_KNN, description = description)
    
    
    # tune K using cross validation
    _, K_optim = knnmodel.cv_tune(X[optimIdxs, :], \
                                  Survival[optimIdxs], \
                                  Censored[optimIdxs], \
                                  kcv = 5, \
                                  shuffles = 5, \
                                  Ks = list(np.arange(10, 160, 10)))
    
    #%%============================================================================
    # Get model accuacy
    #==============================================================================
    
    CIs = knnmodel.cv_accuracy(X, Survival, Censored, \
                               splitIdxs, K = K_optim)

    return CIs

# get accuracy on non-nca-transformed set
CIs_X = get_accuracy(Features)

# get accuracy on nca-transformed set
CIs_XA = get_accuracy(Features_transformed)

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
