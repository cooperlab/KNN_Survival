# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:36:05 2017

@author: mohamed
"""
import os
import numpy as np
from pandas import read_table
import matplotlib.pylab as plt

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
# Get feature ranks
#==============================================================================

site = sites[0]
dtype = dtypes[0]
method = methods[0]
#for site in sites:
#for dtype in dtypes:
#for method in methods:

save_path = base_path + 'Results/tmp/' + site + '_' + dtype + '_' + method.split('_')[0]

# read rank files
ranks_path = result_path + method + '/' + site + '_' + dtype + '_/nca/ranks/'
rank_files = os.listdir(ranks_path)
ranks = []


for fold, rank_file in enumerate(rank_files):
    
    print("Fold {} of {}".format(fold, len(rank_files)-1))
    
    ranks.append(read_table(ranks_path + rank_file, header=None, names=[1]))
    
    fnames = list(ranks[fold].index)
    fnames = [j.split(' ')[0] for j in list(ranks[0].index)]
    ranks[fold].index = fnames

#%%
fweights = []
for fidx, fname in enumerate(fnames):
    
    print("feature {} of {}: {}".format(fidx, len(fnames)-1, fname))
    
    cum_weight = 0
    for fold_ranks in ranks:
        cum_weight += fold_ranks.loc[fname].values[0]
        
    fweights.append(cum_weight)