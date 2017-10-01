# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 01:18:42 2017

@author: mohamed
"""

import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
#sys.path.append('/home/mtageld/Desktop/KNN_Survival/Codes')

from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import SparsePCA as PCA

#%%

projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
#projectPath = "/home/mtageld/Desktop/KNN_Survival/"
RESULTPATH_BASE = projectPath + "Results/6_28Sep2017/"

# dataset and description
site = "GBMLGG"
dtype = "Integ"
description = site +"_"+ dtype +"_"
dpath = projectPath + "Data/SingleCancerDatasets/"+ site+"/"+ \
        site +"_"+ dtype+"_Preprocessed.mat"

Data = loadmat(dpath)
X = Data[dtype + '_X'].copy()
N = X.shape[0]
Survival = Data['Survival'].reshape([N,])
Censored = Data['Censored'].reshape([N,])
Data = None

#%%
pca = PCA()
pca.fit(X)
comp = pca.components_