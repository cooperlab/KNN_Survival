# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:54:33 2017

@author: mohamed
"""

import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')

from scipy.io import loadmat
import numpy as np

import DataManagement as dm
import KNNSurvival as knn

#%%========================================================================
# Prepare inputs
#==========================================================================

print("Loading and preprocessing data.")

# Load data

projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
#projectPath = "/home/mtageld/Desktop/KNN_Survival/"

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

RESULTPATH = projectPath + "Results/tmp/"
MONITOR_STEP = 10
description = "GBMLGG_Gene_"

# remove zero-variance features
fvars = np.std(Features, 0)
keep = fvars > 0
Features = Features[:, keep]
fnames = fnames[keep]

# Get split indices
splitIdxs = dm.get_balanced_SplitIdxs(Censored)

# Get training and testing sets
fold = 10
idxs_train = splitIdxs['fold_cv_train'][fold]
idxs_test = splitIdxs['fold_cv_test'][fold]

X_test = Features[idxs_test, :]
X_train = Features[idxs_train, :]
Survival_train = Survival[idxs_train]
Censored_train = Censored[idxs_train]
Survival_test = Survival[idxs_test]
Censored_test = Censored[idxs_test]

#%%============================================================================
# Train
#==============================================================================

# Instantiate a KNN survival model
knnmodel = knn.SurvivalKNN(RESULTPATH, description = description)

K = 30
idx = 0

#%%
## Predict testing set
#T_test, Ci = knnmodel.predict(X_test, X_train, 
#                              Survival_train, Censored_train, 
#                              Survival_test = Survival_test, 
#                              Censored_test = Censored_test, 
#                              K = 80)


#%%

import matplotlib.pylab as plt

def _plotMonitor(arr, title, xlab, ylab, savename, arr2 = None):
                        
    """ plots cost/other metric to monitor progress """
    
    print("Plotting " + title)
    
    fig, ax = plt.subplots() 
    ax.plot(arr[:,0], arr[:,1], 'b', linewidth=1.5, aa=False)
    if arr2 is not None:
        ax.plot(arr[:,0], arr2, 'r', linewidth=1.5, aa=False)
    plt.title(title, fontsize =16, fontweight ='bold')
    plt.xlabel(xlab)
    plt.ylabel(ylab) 
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()
    
#%%
#
#b = np.concatenate((np.arange(len(status))[:, None], status[:, None]), axis = 1)
#
## get cumproduct
#a = np.cumprod(status)
#
#_plotMonitor(b, '', '', '', '/home/mohamed/Desktop/a.svg', arr2=a)
#
#a2 = a[0:550]
#a2 = np.concatenate((np.arange(len(a2))[:, None], a2[:, None]), axis = 1)
#_plotMonitor(a2, '', '', '', '/home/mohamed/Desktop/a2.svg')
    
    
#%%
    
    
import SurvivalUtils as sUtils

X, T, O, at_risk = sUtils.calc_at_risk(X_train, Survival_train, 1-Censored_train)