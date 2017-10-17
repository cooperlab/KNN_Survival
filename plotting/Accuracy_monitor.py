# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:02:28 2017

@author: mohamed
"""

import os
import numpy as np
import matplotlib.pylab as plt

#%%

base_path = '/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/'
#base_path = '/home/mtageld/Desktop/KNN_Survival/'

result_path = base_path + 'Results/10_10Oct2017/Integ/'

sites = ["GBMLGG", "BRCA", "KIPAN"]
dtypes = ["Integ", "Gene"]
methods = ["cumulative-time_TrueNCA_FalsePCA", "non-cumulative_TrueNCA_FalsePCA"]

#%%

site = sites[0]
dtype = dtypes[0]
method = methods[0]

accuracy_path = result_path + method + '/' + site + '_' + dtype + '_/nca/plots/'

accuracy_files = os.listdir(accuracy_path)

#%%
accuracies_train = [j for j in accuracy_files if 'train' in j]
accuracies_valid = [j for j in accuracy_files if 'valid' in j]

accuracies_train.sort()
accuracies_valid.sort()

fig, axarr = plt.subplots(nrows=2, ncols=2, sharex=True,  sharey=True, figsize=(6, 6))

folds = [0, 6, 16, 24]
fidx = 0

for axrow in range(2):
    for axcol in range(2):
        
        fold = folds[fidx]
        accuracy_train = np.loadtxt(accuracy_path + accuracies_train[fold], delimiter='\t')
        accuracy_valid = np.loadtxt(accuracy_path + accuracies_valid[fold], delimiter='\t')
        
        # plot accuracy change with epoch
        axarr[axrow, axcol].plot(accuracy_train[:, 1], 'b-', linewidth=2)
        axarr[axrow, axcol].plot(accuracy_valid, 'r-', linewidth=2)
        
        # plot snapshot (early stopping)        
        axarr[axrow, axcol].axvline(x= len(accuracy_valid) - 4, linewidth=1.5, color='k', linestyle='--')        
        
        # set axis limits
        axarr[axrow, axcol].set_xlim([0, 25])
        axarr[axrow, axcol].set_ylim([0.7, 0.9])
        
        fidx += 1


plt.subplots_adjust(wspace=0, hspace=0) 
#plt.close()

fig.text(0.5, 0.04, 'Epoch', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'C-index', va='center', rotation='vertical', fontsize=14)

#%%