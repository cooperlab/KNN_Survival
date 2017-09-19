#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat
import _pickle
import sys
sys.path.append("/home/mtageld/Desktop/KNN_Survival/Codes/")
from KNNSurvival import SurvivalKNN as knn

# ===============================================================

# Load data and split indices
dpath = '/home/mtageld/Desktop/KNN_Survival/Data/' + \
    'SingleCancerDatasets/GBMLGG/GBMLGG_Integ_Preprocessed.mat'

data = loadmat(dpath)
with open(dpath.split('.mat')[0] + '_splitIdxs.pkl', 'rb') as f:
    splitIdxs = _pickle.load(f)

N = len(data['Survival'][0])
T = data['Survival'].reshape([N, ])
C = data['Censored'].reshape([N, ])
X = data['Integ_X']

# isolate optimization set for fold
outer_fold = 0
T = T[splitIdxs['idx_optim'][outer_fold]]
C = C[splitIdxs['idx_optim'][outer_fold]]
X = X[splitIdxs['idx_optim'][outer_fold]]

# ==============================================================

RESULTPATH = "/home/mtageld/Desktop/KNN_Survival/Results/tmp/"
subset_size = 10
n_ensembles = 300
K = 100
Method = 'cumulative_time'

model = knn(RESULTPATH)

# ==============================================================


# Generate random ensembles
ensembles = np.random.randint(0, X.shape[1], [n_ensembles, subset_size])

# Initialize accuracy
#CIs = np.zeros(n_ensembles)
feat_ci = np.zeros([n_ensembles, X.shape[1]])

for eidx in range(n_ensembles):

    # get neighbor indices based on this feature ensemble
    fidx = ensembles[eidx, :]
    neighborIdxs = model._get_neighbor_idxs(X[:, fidx], X[:, fidx], norm=2)

    # get accuracy
    _, ci = model.predict(neighborIdxs, T, C,
                          Survival_test=T,
                          Censored_test=C,
                          K=K,
                          Method=Method)
    #CIs[eidx] = ci
    feat_ci[eidx, fidx] = ci

    print("ensemble {} of {}: Ci = {}.".format(eidx, n_ensembles, ci))

# Get feature ranks (lowest first)
feats_sorted = np.argsort(np.sum(feat_ci, axis=0))
featnames_sorted = data['Integ_Symbs'][feats_sorted]
