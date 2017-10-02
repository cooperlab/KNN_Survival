
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 2017

@author: mohamed
"""

#import os
import sys
#sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

import _pickle
from scipy.io import loadmat, savemat
import numpy as np

import DataManagement as dm


def Preprocess_and_split(projectPath, site, dtype, \
                         K=5, SHUFFLES=6, VALID_RATIO=0.25):

    """
    Preprocesses data and splits into optimization and 
    CV folds with shuffles
    """
    
    # Prepare inputs
    #====================================================
    
    dpath = projectPath + "Data/SingleCancerDatasets/"+ site+"/"+ site +"_"+ dtype+".mat"
    #description = site + "_"+ dtype+"_"
    
    # Load data
    Data = loadmat(dpath)
    
    # process features - remove zero-variance
    Data[dtype + '_X'] = np.float32(Data[dtype + '_X'])
    fvars = np.std(Data[dtype + '_X'], 0)
    keep = fvars > 0
    Data[dtype + '_X'] = Data[dtype + '_X'][:, keep]
    
    # process outcomes
    if np.min(Data['Survival']) < 0:
        Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1
    N = Data[dtype + '_X'].shape[0]
    Data['Survival'] = np.int32(Data['Survival']).reshape([N,])
    Data['Censored'] = np.int32(Data['Censored']).reshape([N,])
    
    # Get split indices
    #====================================================
    
    splitIdxs = \
        dm.cv_with_shuffling(N=N, 
                             kcv=K, 
                             n_shuffles=SHUFFLES,
                             valid_ratio=VALID_RATIO)

    # Save
    #====================================================
    
    savename_data = dpath.split('.mat')[0] + "_Preprocessed.mat"
    savename_split = dpath.split('.mat')[0] + "_Preprocessed_splitIdxs.pkl"
   
    savemat(savename_data, Data)

    with open(savename_split,'wb') as f:
        _pickle.dump(splitIdxs, f)


# Peprocess all datasets
#====================================================

if __name__ == '__main__':

    
    #projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
    projectPath = "/home/mtageld/Desktop/KNN_Survival/"
    
    sites = ["GBMLGG", "BRCA", "KIPAN", "MM"]
    dtypes = ["Integ", "Gene"]
    
    for site in sites:
        for dtype in dtypes:

            if ((site == "MM") and (dtype == "Integ")):
                continue

            print("site: {}, dtype: {}".format(site, dtype))

            Preprocess_and_split(projectPath, site, dtype,
                                 K=5,
                                 SHUFFLES=6,
                                 VALID_RATIO=0.25)
