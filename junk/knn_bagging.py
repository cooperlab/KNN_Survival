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

# Load data and split indices
dpath = '/home/mtageld/Desktop/KNN_Survival/Data/' + \
    'SingleCancerDatasets/GBMLGG/GBMLGG_Integ_Preprocessed.mat'

data = loadmat(dpath)
with open(dpath.split('.mat')[0] + '_splitIdxs.pkl', 'rb') as f:
    split_all = _pickle.load(f)

N = len(data['Survival'][0])
Survival = data['Survival'].reshape([N, ])
Censored = data['Censored'].reshape([N, ])
Features = data['Integ_X']

# isolate optimization set
outer_fold = 0
T = Survival[split_all['idx_optim'][outer_fold]]
C = Censored[split_all['idx_optim'][outer_fold]]
X = Features[split_all['idx_optim'][outer_fold]]

# ==============================================================
# Instantiate model / determine params
# ==============================================================

RESULTPATH = "/home/mtageld/Desktop/KNN_Survival/Results/tmp/"

kcv = 4
shuffles = 5
n_ensembles = 25

subset_size = 10
K = 100
Method = 'cumulative_time'
norm = 2

model = knn(RESULTPATH)

# ==============================================================
# Now tune model
# ==============================================================

# Get split indices over optimization set
splitIdxs = \
 dm.get_balanced_SplitIdxs(C,
                           K=kcv, SHUFFLES=shuffles,
                           USE_OPTIM=False)

# Initialize accuracy
n_folds = len(splitIdxs['fold_cv_train'][0])
feat_ci = np.empty((n_ensembles, X.shape[1], n_folds))
feat_ci[:] = np.nan

# Itirate through folds

for fold in range(n_folds):

    # Isolate indices
    train_idxs = splitIdxs['fold_cv_train'][0][fold]
    test_idxs = splitIdxs['fold_cv_test'][0][fold]

    # Generate random ensembles
    ensembles = np.random.randint(0, X.shape[1], [n_ensembles, subset_size])
    
    for eidx in range(n_ensembles):
    
        # get neighbor indices based on this feature ensemble
        fidx = ensembles[eidx, :]
        neighborIdxs = model._get_neighbor_idxs(\
                        X[test_idxs, :][:, fidx], 
                        X[train_idxs, :][:, fidx], 
                        norm=norm)

        # get accuracy
        _, ci = model.predict(\
                 neighborIdxs, 
                 T[train_idxs], 
                 C[train_idxs],
                 Survival_test=T[test_idxs], 
                 Censored_test=C[test_idxs],
                 K=K,
                 Method=Method)

        feat_ci[eidx, fidx, fold] = ci

        print("fold {} of {}, ensemble {} of {}: Ci = {}".\
            format(fold, n_folds-1, eidx, n_ensembles-1, round(ci, 3)))

# Get feature ranks (lowest first)

# median ci across ensembles in each fold
median_ci = np.nanmedian(feat_ci, axis=0)
# median ci accross all folds
median_ci = np.nanmedian(median_ci, axis=1)

feats_sorted = np.argsort(median_ci)
featnames_sorted = data['Integ_Symbs'][feats_sorted]

# ==============================================================
# Get accuracy using top features
# ==============================================================

tune_params = {'kcv': 4,
               'shuffles': 5,
               'Ks': list(np.arange(10, 160, 10)).
               'norm': norm,
               'Method': Method,
               }

CIs, K_optim = model.cv_accuracy(\
                Features[:, feats_sorted[-25:]],
                Survival,
                Censored,
                split_all,
                outer_fold=outer_fold,
                tune_params,
                norm=norm,
                Method=Method)
