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
sites = ["GBMLGG",] # "BRCA", "KIPAN", "LUSC"]
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

# Now run experiment
#=================================================================


USE_NCA = True
Method = Methods[0]

# pass params to dicts
k_tune_params['Method'] = Method
knn_params['Method'] = Method

# Itirate through datasets

RESULTPATH = RESULTPATH_BASE + \
             Method + "_" + \
             str(USE_NCA) + "NCA/"
os.system("mkdir " + RESULTPATH)

dtype = dtypes[0]
site = sites[0]

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
knnmodel = knn.SurvivalKNN(RESULTPATH_KNN, description = description)

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

# Use bagged KNN approach
#=========================

# get accuracy
print("\nOptimizing no of features to use.")

# sort X by absolute feature weights
sort_idxs = np.flip(np.abs(w).argsort(), axis=0)
X = X[:, sort_idxs]

# *** NEEDED INPUTS *** ------

import DataManagement as dm

# norm
# Method
K = 50

X_optim = X[optimIdxs, :]
Survival_optim = Survival[optimIdxs]
Censored_optim = Censored[optimIdxs]

kcv = k_tune_params['kcv']
shuffles = k_tune_params['shuffles']

# ----------------------------

"""
Find the nearest neighbors using the first n_feats. Similar
concept to using the first principal components
of PCA_transformed datasets. 

IMPORTANT: this assumes that the transformed features
have been sorted by andolute feature weight 
(from largest to smallest).

"""

# Get split indices over optimization set
splitIdxs = dm.get_balanced_SplitIdxs(Censored_optim, \
                                      K=kcv, SHUFFLES=shuffles,\
                                      USE_OPTIM = False)

# Initialize
n_folds = len(splitIdxs['fold_cv_train'][0])
CIs = np.zeros([n_folds, len(Ks)])

for fold in range(n_folds):

    print("\n\tFold {} of {}".format(fold, n_folds-1))
    
    # Isolate patients belonging to fold

    idxs_train = splitIdxs['fold_cv_train'][0][fold]
    idxs_test = splitIdxs['fold_cv_test'][0][fold]
    
    X_test = X[idxs_test, :]
    X_train = X[idxs_train, :]
    Survival_train = Survival[idxs_train]
    Censored_train = Censored[idxs_train]
    Survival_test = Survival[idxs_test]
    Censored_test = Censored[idxs_test]

    print("\tn_feats\tCi")

    for n_feats in range(1, 100): #X.shape[1]):

        # Get neighbor indices
        neighbor_idxs = knnmodel._get_neighbor_idxs(\
                X_test[:, 0:n_feats], 
                X_train[:, 0:n_feats], 
                norm=norm)
    
        # Predict testing set
        _, Ci = knnmodel.predict(neighbor_idxs,
                                 Survival_train, Censored_train, 
                                 Survival_test = Survival_test, 
                                 Censored_test = Censored_test, 
                                 K = K, Method = Method)



        CIs[fold, n_feats] = Ci
        
        print("\t{} \t {}".format(K, round(Ci, 3)))


# Get optimal K
CIs_median = np.median(CIs, axis=0)
CI_optim = np.max(CIs_median)
n_feats_optim = np.argmax(CIs_median)+1
print("\nOptimal: n_feats = {}, Ci = {}\n".format(K_optim, round(CI_optim, 3)))




#====================================================================
#====================================================================

#ci, _ = knnmodel.cv_accuracy(X, Survival, Censored, \
#                             splitIdxs, outer_fold=outer_fold,\
#                             tune_params=k_tune_params)
#CIs[:, outer_fold] = ci
#   
#
#print("\nAccuracy")
#print("------------------------")
#print("25th percentile = {}".format(np.percentile(CIs, 25)))
#print("50th percentile = {}".format(np.percentile(CIs, 50)))
#print("75th percentile = {}".format(np.percentile(CIs, 75)))
#
#
## Save results
#print("\nSaving final results.")
#with open(RESULTPATH + description + 'testing_Ci.txt','wb') as f:
#    np.savetxt(f, CIs, fmt='%s', delimiter='\t')
