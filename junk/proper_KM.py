#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 01:56:26 2017

@author: mohamed

"""

import numpy as np
from scipy.io import loadmat

#%%========================================================================
# Prepare inputs
#==========================================================================

print("Loading and preprocessing data.")

# Load data

#projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
projectPath = "/home/mtageld/Desktop/KNN_Survival/"

#dpath = projectPath + "Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
#dpath = projectPath + "Data/SingleCancerDatasets/GBMLGG/Brain_Gene.mat"
dpath = projectPath + "Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat"

description = "BRCA_Integ_"

Data = loadmat(dpath)

Features = np.float32(Data['Integ_X'])
#Features = np.float32(Data['Gene_X'])

N, D = Features.shape

if np.min(Data['Survival']) < 0:
    Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1

Survival = np.int32(Data['Survival']).reshape([N,])
Censored = np.int32(Data['Censored']).reshape([N,])


#%%==========================================

# generate synthetic data
#T = np.random.randint(0, 300, [150,])
#C = np.random.binomial(1, 0.3, [150,])
T = Survival
C = Censored


# find unique times
t = np.unique(T[C == 0])

# initialize count vectors
f = np.zeros(t.shape)
d = np.zeros(t.shape)
n = np.zeros(t.shape)

# generate counts
for i in range(len(t)):
    n[i] = np.sum(T >= t[i])
    d[i] = np.sum(T[C == 0] == t[i])

# calculate probabilities
f = (n - d) / n
f = np.cumprod(f)

# append beginning and end points
t_start = np.array([0])
f_start = np.array([1])
t_end = np.array([T.max()])

t = np.concatenate((t_start, t, t_end), axis=0)
f = np.concatenate((f_start, f), axis=0)

# integrate
p = np.sum(np.diff(t) * f)
