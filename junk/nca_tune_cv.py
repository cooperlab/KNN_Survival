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

import NCA_model as nca
import KNNSurvival as knn

#%%

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

# Now run experiment
#=================================================================


USE_NCA = True
Method = Methods[0]
#for USE_NCA in [True, False]:
#    for Method in Methods:

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

#if success != 0:
#    print("Folder exists, experiment already done.")
#    continue

dtype = "Integ"
site = "GBMLGG"
#for dtype in dtypes:
#    for site in sites:

if dtype == "Gene":
    nca_train_params['BATCH_SIZE'] = 40
    nca_train_params['MAX_ITIR'] = 3
else:
    nca_train_params['BATCH_SIZE'] = 400
    nca_train_params['MAX_ITIR'] = 50

#if (site == "MM") and (dtype == "Integ"):
#    continue


description = site +"_"+ dtype +"_"
dpath = projectPath + "Data/SingleCancerDatasets/"+ site+"/"+ \
        site +"_"+ dtype+"_Preprocessed.mat"

# create output directory
os.system('mkdir ' + RESULTPATH + description)


#%%

#============================================
# *** get_cv_accuracy <- method ***
#============================================

# Args !!!!!!!!!!!!!!!

RESULTPATH=RESULTPATH + description + '/'

# !!!!!!!!!!!!!!!!!!!!

print("\n--------------------------------------")
print("Getting cv accuracy: {}, {}".format(site, dtype))
print("--------------------------------------\n")

print("Loading data.")

Data = loadmat(dpath)

Features = Data[dtype + '_X']

if site != "MM":
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

outer_fold = 0
#for outer_fold in range(n_outer_folds):
    
print("\nOuter fold {} of {}\n".format(outer_fold, n_outer_folds-1))

# isolate features
# Note, this is done since they will
# be modified locally in each outer loop
X = Features.copy()

# Isolate optimization set 
optimIdxs = splitIdxs['idx_optim'][outer_fold]

USE_NCA = True
#if USE_NCA:

# instantiate NCA model
ncamodel = nca.SurvivalNCA(RESULTPATH_NCA, \
                           description = description, \
                           LOADPATH = LOADPATH)
                           
    
#%%
# Finding optimal values for ALPHA and LAMBDA (regularization)
#==============================================================

# Params !!!!!!!!!!!!!!!!!!!!
VALID_RATIO = 0.5
K = 50
# !!!!!!!!!!!!!!!!!!!!!!!!!!!

ALPHAS = np.arange(0, 1.1, 0.2)
LAMBDAS = np.arange(0.2, 1.1, 0.2)

# Get training and validation set
stoppoint = int(VALID_RATIO * len(optimIdxs))
optimIdxs_valid = optimIdxs[0:stoppoint]
optimIdxs_train = optimIdxs[stoppoint:]

x_valid = X[optimIdxs_valid, :]

cis = []

for ALPHA in ALPHAS:
    for LAMBDA in LAMBDAS:

        graphParams['ALPHA'] = ALPHA
        graphParams['LAMBDA'] = LAMBDA
        
        w = ncamodel.train(features = X[optimIdxs_train, :],\
                           survival = Survival[optimIdxs_train],\
                           censored = Censored[optimIdxs_train],\
                           COMPUT_GRAPH_PARAMS = graphParams,\
                           **nca_train_params)
        ncamodel.reset_TrainHistory()
        
        # transform
        W = np.zeros((len(w), len(w)))
        np.fill_diagonal(W, w)
        x_valid_transformed = np.dot(X[optimIdxs_valid, :], W)
        
        # get neighbor indices    
        neighbor_idxs = knnmodel._get_neighbor_idxs(x_valid_transformed, 
                                                    X[optimIdxs_train, :], 
                                                    norm = norm)
        
        # Predict testing set
        _, Ci = knnmodel.predict(neighbor_idxs,
                                 Survival_train=Survival[optimIdxs_train], 
                                 Censored_train=Censored[optimIdxs_train], 
                                 Survival_test = Survival[optimIdxs_valid], 
                                 Censored_test = Censored[optimIdxs_valid], 
                                 K = K, Method = Method)
        
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
    
raise Exception("On purpose.")

# Learn NCA matrix on optimization set
#========================================

print("\nLearning NCA on optimization set\n")



graphParams['ALPHA'] = 1
graphParams['LAMBDA'] = 0.1

w = ncamodel.train(features = X[optimIdxs, :],\
                   survival = Survival[optimIdxs],\
                   censored = Censored[optimIdxs],\
                   COMPUT_GRAPH_PARAMS = graphParams,\
                   **nca_train_params)
 

# Transform features according to learned nca model
#===================================================

print("\nTransforming feats according to learned NCA model.")

# transform
#X = np.dot(X, W)
