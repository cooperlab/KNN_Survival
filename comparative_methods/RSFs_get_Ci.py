# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:32:30 2017

@author: mohamed
"""

import os
import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes/')

import numpy as np
from pandas import DataFrame as df, read_table

#%%============================================================================
# Read RSF predictions
#==============================================================================

base_path = '/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/9_8Oct2017/'

n_folds = 30
idxNames = ['site', 'dtype']
folds = list(np.arange(n_folds))
folds = ['fold_' + str(j) for j in folds]
idxNames.extend(folds)
idxNames.extend(['median', 'mean', '25%ile', '75%ile', 'stdev'])

CIs = df(index=idxNames)

print("\nsite\tdtype")
print("-------------------------------")

site = "GBMLGG"
dtype = "Integ"
#for site in ["GBMLGG", "KIPAN"]:
#    for dtype in ["Integ", "Gene"]:

print(site + "\t" + dtype)

pred_path = base_path + site + '_' + dtype + '/'

result_files = os.listdir(pred_path)
val_files = [j for j in result_files if 'preds_val' in j]
test_files = [j for j in result_files if 'preds_test' in j]

fold = 0
# for fold in folds

preds_val = [read_table(pred_path + j, sep=' ') for j in val_files]
preds_test = [read_table(pred_path + j, sep=' ') for j in test_files]