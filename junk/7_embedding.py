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

#%%============================================================================
# Define params
#==============================================================================

base_path = '/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/'
#base_path = '/home/mtageld/Desktop/KNN_Survival/'
result_path = base_path + 'Results/12_21Oct2017/'

sites = ["GBMLGG", "KIPAN"]
dtypes = ["Integ", ] # "Gene"]
methods = ["cumulative-time_TrueNCA_FalsePCA", "non-cumulative_TrueNCA_FalsePCA"]

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
Survival = Data['Survival'].reshape([N,])
Censored = Data['Censored'].reshape([N,])
fnames = Data[dtype + '_Symbs']
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

# Fetch accuracies and sort
accuracy_path = specific_path + 'nca/plots/'
accuracy_files = os.listdir(accuracy_path)
accuracy_files = [j for j in accuracy_files if 'valid' in j]
accuracy_files.sort()

#%% 
# Fetch embeddings and plot
#==============================================================================

embed_idx = 16; embed_fname = embedding_files[embed_idx]

#for embed_idx, embed_fname in enumerate(embedding_files):

embedding = np.load(embed_path + embed_fname)

ci_valid = np.loadtxt(accuracy_path + accuracy_files[embed_idx])


#%%
# plot and save
# -------------------------------------------------------------------------

plt.scatter(embedding[:, 0], embedding[:, 1])

plt.title("ci_valid = {}, n_epochs = {}".format(round(ci_valid[-1], 3), len(ci_valid)), fontsize=16)
plt.savefig(result_path + '/tmp/' + embed_fname + '.svg')
plt.close()