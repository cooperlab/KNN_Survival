# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:36:05 2017

@author: mohamed
"""

#import sys
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

# sort rank files so first folds come first
# note that the time is in file name so this sorts them well
rank_files.sort()

ranks = []


for fold, rank_file in enumerate(rank_files):
    
    print("Fold {} of {}".format(fold, len(rank_files)-1))
    
    rnk = read_table(ranks_path + rank_file, header=None, names=[1])
    fnames = list(rnk.index)
    fnames = [j.split(' ')[0] for j in list(rnk.index)]
    
    ranks.append(fnames)


#%%============================================================================
# Aggregate ranks from various folds
#==============================================================================


feature_ranks = np.zeros((len(fnames), len(ranks)))

for fidx, fname in enumerate(fnames):
    
    # print("feature {} of {}: {}".format(fidx, len(fnames)-1, fname))
    
    this_feature_ranks = []
    
    for fold, fold_ranks in enumerate(ranks):
        feature_ranks[fidx, fold] = fold_ranks.index(fname) + 1
        
    # sort so that lower-ranked folds appear first (for this feature)
    feature_ranks[fidx, :].sort()

# no of "informative"/less-noisy folds to use
n_top_folds = 10

# Get ranks and sort
overall_ranks = np.mean(feature_ranks[:, 0:n_top_folds], axis=1)
sorted_idxs = np.argsort(overall_ranks)
overall_ranks = overall_ranks[sorted_idxs]
feature_ranks = feature_ranks[sorted_idxs, :]
fnames_sorted = np.array(fnames)[sorted_idxs]


#%%============================================================================
# Plotting
#==============================================================================

# number of top features to plot
n_top = 30

plt.plot(overall_ranks[0:n_top], linewidth=2, color='r', linestyle='-')

# scatter fold ranks
for fidx in range(n_top):
    plt.scatter(x=fidx * np.ones((n_top_folds,)), 
                y=feature_ranks[fidx, 0:n_top_folds], 
                alpha=0.4, color='grey')

# Add top feature names                
for i, txt in enumerate(fnames_sorted[0:n_top]):
    
    if len(txt) > 20:
        txt = txt.split('-')[-1]
    txt = txt.replace('_', ' ')
    
    # color code different data types
    if 'Mut' in txt:
        col = 'r'
    elif 'CNVArm' in txt:
        col = 'purple'
    elif 'CNV' in txt:
        col = 'b'
    else:
        col = 'k'
    
    # now add text annotations
    plt.annotate(txt.replace('_', ' '), (i - 0.5, 10 + overall_ranks[i]), 
                 va= 'bottom',
                 rotation= 'vertical',
                 fontsize= 9,
                 color=col)
                 
# fix ylabel and range
plt.ylabel("Rank", fontsize=14)
plt.ylim(ymin=0, ymax=50)

# remove x ticks (color coded instead)
plt.xlim(xmin=-1, xmax=n_top)
plt.xticks(np.arange(n_top), [])

# plot title
plt.title(site + ' ' + dtype, fontsize=16, fontweight='bold')

# Save
plt.savefig(save_path + '_featranks.svg', figsize=(6, 6), bbox_inches='tight')
plt.close()