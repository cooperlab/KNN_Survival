#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:11:40 2017

@author: mtageld
"""

import numpy as np
from pandas import DataFrame as df

#%%

resultpath_base = "/home/mtageld/Desktop/KNN_Survival/Results/5_27Sep2017/"

# rownames
n_folds = 42
idxNames = ['method', 'PCA', 'NCA', 'site', 'dtype']
folds = list(np.arange(n_folds))
folds = ['fold_' + str(j) for j in folds]
idxNames.extend(folds)
idxNames.extend(['median', 'mean', '25%ile', '75%ile', 'stdev'])

CIs = df(index=idxNames)

#%%
for site in ["GBMLGG", "BRCA", "KIPAN", "MM"]:
    for dtype in ["Integ", "Gene"]:
        for method in ["cumulative-time", "non-cumulative"]:
            for PCA in ["False", "True"]:
                for NCA in ["False", "True"]:
                    
                    if ((dtype == "Integ") and (site == "MM")):
                        continue
                    if ((dtype == "Gene") and (PCA == "False")):
                        continue
                        
                    resultpath = resultpath_base + method + "_" + NCA + "NCA_" + PCA + "PCA/" + \
                                 site + "_" + dtype + "_/" + site + "_" + dtype + "_testing_Ci.txt"
                    
                    ci = np.loadtxt(resultpath, delimiter='\t').reshape([n_folds,])
                    ci_merge = [method, PCA, NCA, site, dtype]
                    ci_merge.extend(ci)
                    ci_merge.extend([np.median(ci), np.mean(ci), \
                                     np.percentile(ci, 25), np.percentile(ci, 75), \
                                     np.std(ci)])
                    
                    
                    CIs[CIs.shape[1]] = ci_merge
        
#%%

CIs.to_csv(resultpath_base + "results_merged.tab", sep="\t")
