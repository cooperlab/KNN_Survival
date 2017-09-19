#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat

import sys
sys.path.append("/home/mtageld/Desktop/KNN_Survival/Codes/")
from KNNSurvival import SurvivalKNN as knn

#===============================================================

dpath = '/home/mtageld/Desktop/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/GBMLGG_Integ_Preprocessed.mat'

data = loadmat(dpath)

N = len(data['Survival'][0])
T = data['Survival'].reshape([N,])
C = data['Censored'].reshape([N,])
X = data['Integ_X']

# PLAYING AROUND ********
T = T[0:100]
C = C[0:100]
X = X[0:100, :]

#==============================================================

RESULTPATH = "/home/mtageld/Desktop/KNN_Survival/Results/tmp"
model = knn(RESULTPATH)

neighborIdxs = model._get_neighbor_idxs(X, X, norm=2)

subset_size = 100
fidx = np.random.randint(0, X.shape[1], [subset_size,])

neighborIdxs_1 = model._get_neighbor_idxs(X[:, fidxs], X[:, fidxs])
