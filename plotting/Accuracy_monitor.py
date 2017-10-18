# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:02:28 2017

@author: mohamed
"""

import os
import numpy as np
import matplotlib.pylab as plt

#%%============================================================================
# Plot cost change (method)
#==============================================================================

def plot_cost_change(accuracy_path, save_path, folds,
                     maxfolds=25, ci_min=0.7, ci_max=0.9, 
                     model_buffer=4):
    
    """
    Plots change in accuracy/cost as training progresses    
    """

    # read accuracy files
    accuracy_files = os.listdir(accuracy_path)
    costs_train = [j for j in accuracy_files if 'costs' in j]
    accuracies_train = [j for j in accuracy_files if 'train' in j]
    accuracies_valid = [j for j in accuracy_files if 'valid' in j]
    
    costs_train.sort()
    accuracies_train.sort()
    accuracies_valid.sort()
    
    # Initialize fig
    n_folds  = len(folds)
    fig, axarr = plt.subplots(nrows=n_folds, ncols=2, figsize=(6, 6))
    
    fidx = 0
    
    for axrow in range(n_folds):
        
        # plot costs/Ci
        fold = folds[fidx]
        cost_train = np.loadtxt(accuracy_path + costs_train[fold], delimiter='\t')
        accuracy_train = np.loadtxt(accuracy_path + accuracies_train[fold], delimiter='\t')
        accuracy_valid = np.loadtxt(accuracy_path + accuracies_valid[fold], delimiter='\t')
        
        # plot accuracy change with epoch
        axarr[axrow, 0].plot(cost_train[:, 1], 'b-', linewidth=2)
        axarr[axrow, 1].plot(accuracy_train[:, 1], 'b-', linewidth=2)
        axarr[axrow, 1].plot(accuracy_valid, 'r-', linewidth=2)
        
        # plot snapshot (early stopping)        
        axarr[axrow, 0].axvline(x= len(accuracy_valid) - model_buffer, 
                                linewidth=1.5, color='k', linestyle='--')  
        axarr[axrow, 1].axvline(x= len(accuracy_valid) - model_buffer, 
                                linewidth=1.5, color='k', linestyle='--')        
        
        # set axis limits
        axarr[axrow, 0].set_xlim([0, maxfolds])
        axarr[axrow, 1].set_xlim([0, maxfolds])
        axarr[axrow, 1].set_ylim([ci_min, ci_max])
        
        # increment
        fidx += 1
    
    # Add common title and axis labels
    fig.text(0.5, 0.94, 'Cost/C-index change', 
             ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.05, 'Epoch', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'Cost (left) / C-index (right)', 
             va='center', rotation='vertical', fontsize=14)
    
    # save and close
    plt.savefig(save_path + '_cost_change.svg')
    plt.close()

#%% ###########################################################################
#%% ###########################################################################
#%% ###########################################################################

if __name__ == '__main__':

    #%%============================================================================
    # Ground work
    #==============================================================================
    base_path = '/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/'
    #base_path = '/home/mtageld/Desktop/KNN_Survival/'
    
    result_path = base_path + 'Results/10_10Oct2017/Integ/'
    
    sites = ["GBMLGG", "KIPAN"]
    dtypes = ["Integ", ] #"Gene"]
    methods = ["cumulative-time_TrueNCA_FalsePCA", "non-cumulative_TrueNCA_FalsePCA"]
    
    #%%============================================================================
    # Plot accuracy monitor
    #==============================================================================
    
    for site in sites:
        for dtype in dtypes:
            for method in methods:
            
                accuracy_path = result_path + method + '/' + site + '_' + dtype + '_/nca/plots/'
                save_path = base_path + 'Results/tmp/' + site + '_' + dtype + '_' + method
                
                if site == 'GBMLGG':
                    folds = [6, 16] #[0, 6, 16, 24]
                    plot_params = {'maxfolds': 25, 'ci_min': 0.7, 'ci_max': 0.9}
                    
                elif site == 'KIPAN':
                    folds =  [0, 19] #[0, 7, 15, 19] # KIPAN
                    plot_params = {'maxfolds': 16, 'ci_min': 0.6, 'ci_max': 0.8}
                
                else:
                    raise Exception('no specified folds for site')    
                
                # Now plot and save    
                plot_cost_change(accuracy_path, save_path, folds, **plot_params)