# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:00:11 2017

@author: mohamed
"""

import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
#sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

import os
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
#from scipy.stats import spearmanr
import _pickle

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

#%%============================================================================
# Get data and split indices
#==============================================================================

dpath = base_path + "Data/SingleCancerDatasets/"+ site+"/" + \
        site +"_"+ dtype+"_Preprocessed.mat"

print("Loading data and split indices.")

# Get data
Data = loadmat(dpath)
Features = Data[dtype + '_X'].copy()
N = Features.shape[0]
P = Features.shape[1]
Survival = Data['Survival'].reshape([N,])
Censored = Data['Censored'].reshape([N,])
fnames = Data[dtype + '_Symbs']
fnames = [j.split(' ')[0] for j in fnames]
Data = None

# Get relevant split indices
with open(dpath.split('.mat')[0] + '_splitIdxs.pkl','rb') as f:
        splitIdxs = _pickle.load(f)

#%%============================================================================
# Get result files
#==============================================================================

specific_path = result_path + method + '/' + site + '_' + dtype + '_' + '/'

# Fetch testing Cis and get sorting
CIs_test = np.loadtxt(specific_path + site + '_' + dtype + '_testing_Ci.txt')
top_folds = np.argsort(CIs_test)[::-1]


# Fetch corresponding training/validation Cis
accuracy_path = specific_path + 'nca/plots/'
accuracy_files = os.listdir(accuracy_path)
accuracy_files.sort()

# training Cis
CIs_train = [np.loadtxt(accuracy_path + j)[:, 1] for j in accuracy_files if 'train' in j]
CIs_valid = [np.loadtxt(accuracy_path + j) for j in accuracy_files if 'valid' in j]

#%%============================================================================
# Q0 - Is it an overfitting issue?
#==============================================================================

n_folds = len(CIs_test)
n_bottom_folds = 5

for idx in range(1, n_bottom_folds+1):
    
    foldrank = n_folds - idx 

    plt.plot(CIs_train[top_folds[-idx]], linewidth=2, c='b', linestyle='-')
    plt.plot(CIs_valid[top_folds[-idx]], linewidth=2, c='r', linestyle='-')
    plt.axhline(CIs_test[top_folds[-idx]], linewidth=2, c='r', linestyle='--')
    
    plt.title("fold rank = {} of {}".format(foldrank, n_folds-1), fontsize=16)
    plt.savefig(result_path + '/tmp/' + str(foldrank) + '.svg')
    plt.close()
