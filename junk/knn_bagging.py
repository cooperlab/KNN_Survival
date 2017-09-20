#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat
import _pickle
import sys
sys.path.append("/home/mtageld/Desktop/KNN_Survival/Codes/")
from KNNSurvival import SurvivalKNN as knn
import DataManagement as dm

# ===============================================================
# Load optimization set
# ===============================================================

site = "GBMLGG"
dtype = "Integ"

# Load data and split indices
dpath = '/home/mtageld/Desktop/KNN_Survival/Data/' + \
    'SingleCancerDatasets/'+site+'/'+site+'_'+dtype+'_Preprocessed.mat'

data = loadmat(dpath)
with open(dpath.split('.mat')[0] + '_splitIdxs.pkl', 'rb') as f:
    split_all = _pickle.load(f)

N = len(data['Survival'][0])
Survival = data['Survival'].reshape([N, ])
Censored = data['Censored'].reshape([N, ])
Features = data[dtype + '_X']

# isolate optimization set
outer_fold = 0
T = Survival[split_all['idx_optim'][outer_fold]]
C = Censored[split_all['idx_optim'][outer_fold]]
X = Features[split_all['idx_optim'][outer_fold]]

# ==============================================================
# Instantiate model / determine params
# ==============================================================

RESULTPATH = "/home/mtageld/Desktop/KNN_Survival/Results/tmp/"
Method = 'non-cumulative'
norm = 2

ensemble_params = {'featnames': data[dtype+'_Symbs'], 
                   'kcv': 4,
                   'shuffles': 5,
                   'n_ensembles': 25,
                   'subset_size': 30,
                   'K': 100,
                   'Method': Method,
                   'norm': norm,
                   }

model = knn(RESULTPATH)

# ==============================================================
# Get top features
# ==============================================================

median_ci, feats_sorted, featnames_sorted = \
    model.ensemble_feat_rank(X, T, C, **ensemble_params)
                                
print("Top 10 features are: \n{}".format(featnames_sorted[0:10]))

# ==============================================================
# Get accuracy using top features
# ==============================================================

n_feats = 25

tune_params = {'kcv': 4,
               'shuffles': 5,
               'Ks': list(np.arange(10, 160, 10)),
               'norm': norm,
               'Method': Method,
               }

CIs, K_optim = model.cv_accuracy(\
                Features[:, feats_sorted[0:n_feats]],
                Survival,
                Censored,
                split_all,
                outer_fold=outer_fold,
                tune_params=tune_params,
                norm=norm,
                Method=Method)

print("\n25% percentile = {}".format(np.percentile(CIs, 25)))
print("50% percentile = {}".format(np.percentile(CIs, 50)))
print("75% percentile = {}".format(np.percentile(CIs, 75)))


