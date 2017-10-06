#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:59:26 2017

@author: mtageld
"""

import numpy as np
#import matplotlib.pylab as plt
from scipy.io import loadmat


PROJECTPATH = "/home/mtageld/Desktop/KNN_Survival/"
RESULTPATH = PROJECTPATH + "Results/8_5Oct2017/"
fweights = np.load(RESULTPATH + 'cumulative-time_TrueNCA_FalsePCA/GBMLGG_Integ_/nca/model/GBMLGG_Integ_featWeights.npy')
fweights = np.diag(fweights)

Data = loadmat(PROJECTPATH + 'Data/SingleCancerDatasets/GBMLGG/GBMLGG_Integ_Preprocessed.mat')
fnames = Data['Integ_Symbs']

#%%

def _plotMonitor(self, arr, title, xlab, ylab, savename, arr2 = None):
                        
    """ plots cost/other metric to monitor progress """
    
#    print("Plotting " + title)
#    
#    fig, ax = plt.subplots() 
#    ax.plot(arr[:,0], arr[:,1], 'b', linewidth=1.5, aa=False)
#    if arr2 is not None:
#        ax.plot(arr[:,0], arr2, 'r', linewidth=1.5, aa=False)
#    plt.title(title, fontsize =16, fontweight ='bold')
#    plt.xlabel(xlab)
#    plt.ylabel(ylab) 
#    plt.tight_layout()
#    plt.savefig(savename)
#    plt.close()
    
    #
    # Saving instead of plotting to avoid
    # Xdisplay issues when using screen
    #
    #print("Saving " + title)
    with open(savename.split('.')[0] + '.txt', 'wb') as f:
        np.savetxt(f, arr, fmt='%s', delimiter='\t')


#%%
def rankFeats(fweights, fnames):
        
        """ ranks features by feature weights or variance after transform"""
        
        D = len(fweights)
        fidx = np.arange(D).reshape(D, 1)        
        
        # rank by feature weight
        ranking_metric = fweights[:, None]        
        ranking_metric = np.concatenate((fidx, ranking_metric), 1)      
    
        # Plot feature weights/variance
        #if D <= 500:
        #    n_plot = ranking_metric.shape[0]
        #else:
        #    n_plot = 500
        #_plotMonitor(ranking_metric[0:n_plot,:], 
        #             "feature weights", 
        #             "feature_index", "weight", 
        #             RESULTPATH + "plots/feat_weights.png")
        
        # rank features
        # sort by absolute weight but keep sign
        ranking = ranking_metric[np.abs(ranking_metric[:,1]).argsort()][::-1]
        
        fnames_ranked = fnames[np.int32(ranking[:,0])].reshape(D, 1)
        fw = ranking[:,1].reshape(D, 1) 
        fnames_ranked = np.concatenate((fnames_ranked, fw), 1)
        
        return fnames_ranked
    
#%%

fnames_ranked = rankFeats(fweights, fnames)