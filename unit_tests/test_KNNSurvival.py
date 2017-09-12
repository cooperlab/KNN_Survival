# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:54:33 2017

@author: mohamed
"""

import sys
#sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

from scipy.io import loadmat
import numpy as np

import DataManagement as dm
import KNNSurvival as knn

#%%========================================================================
# Prepare inputs
#==========================================================================

print("Loading and preprocessing data.")

# Load data

#projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
projectPath = "/home/mtageld/Desktop/KNN_Survival/"

#dpath = projectPath + "Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
dpath = projectPath + "Data/SingleCancerDatasets/GBMLGG/Brain_Gene.mat"
#dpath = projectPath + "Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat"

Data = loadmat(dpath)

#Features = np.float32(Data['Integ_X'])
Features = np.float32(Data['Gene_X'])

N, D = Features.shape

if np.min(Data['Survival']) < 0:
    Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1

Survival = np.int32(Data['Survival']).reshape([N,])
Censored = np.int32(Data['Censored']).reshape([N,])
#fnames = Data['Integ_Symbs']
fnames = Data['Gene_Symbs']

RESULTPATH = projectPath + "Results/tmp/"
MONITOR_STEP = 10
description = "GBMLGG_Gene_"

# remove zero-variance features
fvars = np.std(Features, 0)
keep = fvars > 0
Features = Features[:, keep]
fnames = fnames[keep]

# Get split indices
splitIdxs = dm.get_balanced_SplitIdxs(Censored, OPTIM_RATIO = 0.5,\
                                      OPTIM_TRAIN_RATIO = 0.8,\
                                      K = 3,\
                                      SHUFFLES = 10)

#raise Exception("On purpose.")

#%%============================================================================
# Train
#==============================================================================

# Instantiate a KNN survival model
knnmodel = knn.SurvivalKNN(RESULTPATH, description = description)

#%%

# Range of K's over which to test
Ks = list(np.arange(10, 160, 10))

# Initialize
CIs = np.zeros([len(Ks),])

# Isolate patients belonging to optimization set
X_test = Features[splitIdxs['idx_optim_valid'], :]
X_train = Features[splitIdxs['idx_optim_train'], :]
Survival_train = Survival[splitIdxs['idx_optim_train']]
Censored_train = Censored[splitIdxs['idx_optim_train']]
Survival_test = Survival[splitIdxs['idx_optim_valid']]
Censored_test = Censored[splitIdxs['idx_optim_valid']]

# Get neighbor indices    
neighbor_idxs = knnmodel.get_neighbor_idxs(X_test, X_train)


print("K \t Ci")

for kidx, K in enumerate(Ks):
    
    # Predict testing set
    _, Ci = knnmodel.predict(neighbor_idxs,
                             Survival_train, Censored_train, 
                             Survival_test = Survival_test, 
                             Censored_test = Censored_test, 
                             K = K)
    
    CIs[kidx] = Ci
    
    print("{} \t {}".format(K, round(Ci, 3)))
                     

#%%

#import matplotlib.pylab as plt
#
#def _plotMonitor(arr, title, xlab, ylab, savename, arr2 = None):
#                        
#    """ plots cost/other metric to monitor progress """
#    
#    print("Plotting " + title)
#    
#    fig, ax = plt.subplots() 
#    #ax.plot(arr[:,0], arr[:,1], 'b', linewidth=1.5, aa=False)
#    ax.step(arr[:,0], arr[:,1], 'b', linewidth=1.5, aa=False)
#    if arr2 is not None:
#        #ax.plot(arr[:,0], arr2, 'r', linewidth=1.5, aa=False)
#        ax.step(arr[:,0], arr2, 'r', linewidth=1.5, aa=False)
#    plt.title(title, fontsize =16, fontweight ='bold')
#    plt.xlabel(xlab)
#    plt.ylabel(ylab) 
#    plt.tight_layout()
#    plt.savefig(savename)
#    plt.close()

#%%
    
#import SurvivalUtils as sUtils
#
## Expand dims of AX to [n_samples_test, n_samples_train, n_features], 
## where each "channel" in the third dimension is the difference between
## one testing sample and all training samples along one feature
#dist = X_train[None, :, :] - X_test[:, None, :]
#
## Now get the euclidian distance between
## every patient and all others -> [n_samples, n_samples]
##normAX = tf.norm(normAX, axis=0)
#dist = np.sqrt(np.sum(dist ** 2, axis=2))
#
## Get indices of K nearest neighbors
#neighbor_idxs = np.argsort(dist, axis=1)[:, 0:K]
#
##%%
##
## K-M method
##
#
#idx = 10
#T = Survival_train[neighbor_idxs[idx, :]]
#O = 1 - Censored_train[neighbor_idxs[idx, :]]
#T, O, at_risk, _ = sUtils.calc_at_risk(T, O)
#
#N_at_risk = K - at_risk
#
#P = np.cumprod((N_at_risk - O) / N_at_risk)
#
## plot
#a = np.concatenate((T[:, None], P[:, None]), axis = 1)
#_plotMonitor(a, 'Traditional KM', '', '', '/home/mohamed/Desktop/KM.svg')
#
