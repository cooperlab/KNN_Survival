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

#%%============================================================================
# Train
#==============================================================================

# Instantiate a KNN survival model
knnmodel = knn.SurvivalKNN(RESULTPATH, description = description)

#%%

n_folds = len(splitIdxs['fold_cv_train'])
Ks = list(np.arange(10, 160, 10))

# Initialize
CIs_cum = np.zeros([n_folds, len(Ks)])
CIs_noncum = np.zeros([n_folds, len(Ks)])

for fold in range(n_folds):

    print("\nFold {} of {}\n".format(fold, n_folds))
    print("----------------------------------------")
    
    # Isolate patients belonging to fold

    idxs_train = splitIdxs['fold_cv_train'][fold]
    idxs_test = splitIdxs['fold_cv_test'][fold]
    
    X_test = Features[idxs_test, :]
    X_train = Features[idxs_train, :]
    Survival_train = Survival[idxs_train]
    Censored_train = Censored[idxs_train]
    Survival_test = Survival[idxs_test]
    Censored_test = Censored[idxs_test]

    # Get neighbor indices    
    neighbor_idxs = knnmodel.get_neighbor_idxs(X_test, X_train)
    
    
    print("K \t Ci_cum \t Ci_noncum \t diff")
    
    for kidx, K in enumerate(Ks):
        
        # Predict testing set
        _, Ci_cum = knnmodel.predict(neighbor_idxs,
                                      Survival_train, Censored_train, 
                                      Survival_test = Survival_test, 
                                      Censored_test = Censored_test, 
                                      K = K,
                                      Method = 'cumulative')
        
        # Predict testing set - non-cumulative
        _, Ci_noncum = knnmodel.predict(neighbor_idxs, 
                                      Survival_train, Censored_train, 
                                      Survival_test = Survival_test, 
                                      Censored_test = Censored_test, 
                                      K = K,
                                      Method = 'non-cumulative')
                                      
                                      
        # save and print
        CIs_cum[fold, kidx] = Ci_cum
        CIs_noncum[fold, kidx] = Ci_noncum
        
        
        print("{} \t {} \t {} \t {}".\
              format(K, round(Ci_cum, 3), round(Ci_noncum, 3), 
                     round(Ci_cum - Ci_noncum, 3)))
                     


CIs_diff = CIs_cum - CIs_noncum


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
##%%
##
## Modified KM method
##
#
#alive_train = sUtils.getAliveStatus(Survival_train, Censored_train)
#
#status = alive_train[neighbor_idxs[idx, :], :]
#totalKnown = np.sum(status >= 0, axis = 0)
#status[status < 0] = 0
#
## remove timepoints where there are no known statuses
#status = status[:, totalKnown != 0]
#totalKnown = totalKnown[totalKnown != 0]
#
## get "average" predicted survival time
#status = np.sum(status, axis = 0) / totalKnown
#
## plot
#b = np.concatenate((np.arange(len(status))[:, None], status[:, None]), axis = 1)
#b = b[0:int(np.max(T)), :]
#_plotMonitor(b, 'Modified KM', '', '', '/home/mohamed/Desktop/KM_Modified.svg')