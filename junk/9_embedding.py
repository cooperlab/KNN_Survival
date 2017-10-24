# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:57:35 2017

@author: mohamed
"""

import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
#sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

import os
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
from scipy.stats import spearmanr

#%%============================================================================
# Define params
#==============================================================================

base_path = '/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/'
#base_path = '/home/mtageld/Desktop/KNN_Survival/'
result_path = base_path + 'Results/12_21Oct2017/'

sites = ["GBMLGG", "KIPAN"]
dtypes = ["Integ", ] # "Gene"]
methods = ["cumulative-time_TrueNCA_FalsePCA", "non-cumulative_TrueNCA_FalsePCA"]

n_top_folds = 30
pval_thresh = 0.01
n_feats_to_plot = 10

site = sites[0]
dtype = dtypes[0]
method = methods[0]

#%% 
# Get feature files
#==============================================================================

dpath = base_path + "Data/SingleCancerDatasets/"+ site+"/" + \
        site +"_"+ dtype+"_Preprocessed.mat"

print("Loading data.")
Data = loadmat(dpath)
Features = Data[dtype + '_X'].copy()
N = Features.shape[0]
P = Features.shape[1]
Survival = Data['Survival'].reshape([N,])
Censored = Data['Censored'].reshape([N,])
fnames = Data[dtype + '_Symbs']
fnames = [j.split(' ')[0] for j in fnames]
Data = None

#%% 
# Get result files
#==============================================================================

specific_path = result_path + method + '/' + site + '_' + dtype + '_' + '/'

# Fetch embeddings and sort
embed_path = specific_path + 'nca/model/'
embedding_files = os.listdir(embed_path)
embedding_files = [j for j in embedding_files if '.npy' in j]
embedding_files.sort()
embedding_files = np.array(embedding_files)

# Fetch accuracies and sort
accuracies = np.loadtxt(specific_path + site + '_' + dtype + '_testing_Ci.txt')
top_folds = np.argsort(accuracies)[::-1][0:n_top_folds]

# keep top folds
accuracies = accuracies[top_folds]
embedding_files = embedding_files[top_folds]

#sys.exit()

#%%============================================================================
# Itirate through embeddings and get cluster separations
#==============================================================================

threshold = 0 # Z-scored

NC_deltas = np.zeros((len(embedding_files), P))

#embed_idx = 16; embed_fname = embedding_files[embed_idx]
for embed_idx, embed_fname in enumerate(embedding_files):
    
    print("fold {}".format(embed_idx))

    # Get embedding
    embedding = np.dot(Features, np.load(embed_path + embed_fname))
    
    # Itirate through features and get cluster separation
    for fidx in range(P):
    
        # Get cluster separation
        feat_is_present = 0 + (Features[:, fidx] > threshold)
        
        # Get distance across various neighborhood components
        NC_delta = 0
        for ncidx in range(embedding.shape[1]):
            delta = np.mean(embedding[feat_is_present==0, ncidx]) - \
                    np.mean(embedding[feat_is_present==1, ncidx])    
            NC_delta = NC_delta + delta**2
        
        NC_deltas[embed_idx, fidx] = np.sqrt(NC_delta)

#%%============================================================================
# Rank features by correlation between their separation and accuracy
#==============================================================================

rhos = np.zeros((P, ))
pvals = np.zeros((P, ))

# find spearman correlations
for fidx in range(P):
    corr = spearmanr(accuracies, NC_deltas[:, fidx])
    rhos[fidx] = corr[0]
    pvals[fidx] = corr[1]

rhos_signif = rhos
rhos_signif[pvals > pval_thresh] = 0

top_feat_idxs = np.argsort(rhos_signif)[::-1]
top_feat_names = np.array(fnames)[top_feat_idxs]

#%%============================================================================
# Visualize top features
#==============================================================================

#fidx = top_feat_idxs[0]
for fidx in top_feat_idxs[0:n_feats_to_plot]:
    
    print("plotting " + fnames[fidx])

    # scatter points
    plt.scatter(accuracies, NC_deltas[:, fidx])
    
    # plot line of best fit
    slope, intercept = np.polyfit(accuracies, NC_deltas[:, fidx], deg=1)
    abline_values = [slope * i + intercept for i in accuracies]
    plt.plot(accuracies, abline_values, 'b--')
    
    pval_string = round(pvals[fidx], 3)
    if pval_string == 0:
        pval_string = '< 0.001'
    
    fname_string = fnames[fidx].replace('_', ' ')[0: 15]
    
    plt.title(fname_string + ": spRho= {}, p {}".\
              format(round(rhos[fidx], 3), pval_string), fontsize=16)
    plt.xlabel("Testing C-index", fontsize=14)
    plt.ylabel("cluster separation", fontsize=14)
    plt.savefig(result_path + '/tmp/' + fnames[fidx] + '_corr.svg')
    plt.close()


#%%
# Mutations
#------------------------------------------------------------------------------

# IDHwt
#is_IDHwt = isolate_feats(fnames=("IDH1_Mut", "IDH2_Mut"), thresholds=(0, 0))
is_IDHwt = isolate_feats(fnames=("IDH1_Mut", ), thresholds=(0, ))
is_IDHwt = 1 - np.int32(is_IDHwt > 0)

# CIC
is_CIC = isolate_feats(fnames=("CIC_Mut",), thresholds=(0,))
is_CIC = np.int32(is_CIC > 0)


#%%
# Fetch embedding and plot
#==============================================================================

NC_delta = np.zeros([len(embedding_files,)])    

#embed_idx = 16; embed_fname = embedding_files[embed_idx]
for embed_idx, embed_fname in enumerate(embedding_files):

    print("fold {}".format(embed_idx))
    
    # Get embedding
    embedding = np.dot(Features, np.load(embed_path + embed_fname))
    
    # Get IDH cluster separation
    NC0_delta = np.mean(embedding[is_IDHwt==0, 0]) - np.mean(embedding[is_IDHwt==1, 0])
    NC1_delta = np.mean(embedding[is_IDHwt==0, 1]) - np.mean(embedding[is_IDHwt==1, 1])
    NC_delta[embed_idx] = np.sqrt(NC0_delta**2 + NC1_delta**2)
    
    # plot and save

    # IDH only
    plt.scatter(embedding[is_IDHwt==0, 0], embedding[is_IDHwt==0, 1], c='b')
    plt.scatter(embedding[is_IDHwt==1, 0], embedding[is_IDHwt==1, 1], c='r')
    plt.title("IDH - ci_test = {}, NC_delta = {}".\
               format(round(accuracies[embed_idx], 3), 
                      round(NC_delta[embed_idx], 3)), 
               fontsize=16)
    plt.xlabel("NC1", fontsize=14)
    plt.ylabel("NC2", fontsize=14)
    plt.savefig(result_path + '/tmp/IDH/' + embed_fname.split('.npy')[0] + '.svg')
    plt.close()
    
    # CIC only
    plt.scatter(embedding[is_CIC==0, 0], embedding[is_CIC==0, 1], c='k')
    plt.scatter(embedding[is_CIC==1, 0], embedding[is_CIC==1, 1], c='gold')
    plt.title("CIC - ci_test = {}".format(round(accuracies[embed_idx], 3)), fontsize=16)
    plt.xlabel("NC1", fontsize=14)
    plt.ylabel("NC2", fontsize=14)
    plt.savefig(result_path + '/tmp/CIC/' + embed_fname.split('.npy')[0] + '.svg')
    plt.close()
    
    # IDH1 and CIC
    plt.scatter(embedding[is_IDHwt==0, 0], embedding[is_IDHwt==0, 1], c='k')
    plt.scatter(embedding[is_CIC==1, 0], embedding[is_CIC==1, 1], c='gold')
    plt.scatter(embedding[is_IDHwt==1, 0], embedding[is_IDHwt==1, 1], c='r')
    plt.title("IDH-CIC - ci_test = {}, NC_delta = {}".\
               format(round(accuracies[embed_idx], 3), 
                      round(NC_delta[embed_idx], 3)), 
               fontsize=16)
    plt.xlabel("NC1", fontsize=14)
    plt.ylabel("NC2", fontsize=14)
    plt.savefig(result_path + '/tmp/IDH-CIC/' + embed_fname.split('.npy')[0] + '.svg')
    plt.close()

# find spearman correlation
rho, pval = spearmanr(accuracies, NC_delta)
            
# scatter points
plt.scatter(accuracies, NC_delta)

# plot line of best fit
slope, intercept = np.polyfit(accuracies, NC_delta, deg=1)
abline_values = [slope * i + intercept for i in accuracies]
plt.plot(accuracies, abline_values, 'b--')

plt.title("IDH1 - spearman rho = {}, p = {}".format(round(rho, 3), round(pval, 4)), fontsize=16)
plt.xlabel("Testing C-index", fontsize=14)
plt.ylabel("cluster separation", fontsize=14)
plt.savefig(result_path + '/tmp/IDH' + '.svg')
plt.close()