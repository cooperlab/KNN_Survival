#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 01:56:26 2017

@author: mohamed

Survival NCA (Neighborhood Component Analysis)
"""

# Append relevant paths
import os
import sys

def conditionalAppend(Dir):
    """ Append dir to sys path"""
    if Dir not in sys.path:
        sys.path.append(Dir)

cwd = os.getcwd()
conditionalAppend(cwd)

import numpy as np
from scipy.io import loadmat, savemat
import scipy.optimize as opt

import SurvivalUtils as sUtils
import nca_cost


#raise Exception()

#%%============================================================================
# ---- J U N K ----------------------------------------------------------------
#==============================================================================

# Load data
dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
Data = loadmat(dpath)

data = np.float32(Data['Integ_X'])
if np.min(Data['Survival']) < 0:
    Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1

Survival = np.int32(Data['Survival'])
Censored = np.int32(Data['Censored'])
fnames = Data['Integ_Symbs']

# Limit patient no for prototyping
n = 50
data = data[0:n,:]
Survival = Survival[0:n,:]
Censored = Censored[0:n,:]

# remove zero-variance features
fvars = np.std(data, 0)
keep = fvars > 0
data = data[:, keep]
fnames = fnames[keep]

# Generate survival status - discretized into months
aliveStatus = sUtils.getAliveStatus(Survival, Censored, scale = 30)


#%%============================================================================
# --- P R O T O T Y P E S -----------------------------------------------------
#==============================================================================

RESULTPATH = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/tmp/"

DIMS = 100 #data.shape[1] # set DIMS < D to reduce dimensions

OBJECTIVE = 'Mahalanobis'
THRESH = None # None or between [0, 1], the smaller the more emphasis on closer neighbors
#MAXITER = 100
MONITOR_STEP = 10



#%%============================================================================
# Setting things up
#==============================================================================

# Get dims
N, D = data.shape
T = aliveStatus.shape[1] # no of time points

# Initialize A to a scaling matrix
A = np.eye(D, DIMS)


#%%============================================================================
# Go through time points
#==============================================================================

# initialize cum_f and cum_gradf
cum_f = 0
cum_gradf = np.zeros(A.shape)

#t = 20
for t in range(T):

    print("t = {} of {}".format(t, T-1))
        
    # Get patients with known survival status at time t
    Y = aliveStatus[:, t]
    keep = Y >= 0
    Y = Y[keep]
    X = data[keep, :]
    
    if OBJECTIVE == 'Mahalanobis':
        f, gradf = nca_cost.cost(A.T, X.T, Y, threshold=THRESH)
    elif OBJECTIVE == 'KL-divergence':
        f, gradf = nca_cost.cost_g(A.T, X.T, Y, threshold=THRESH)
        
    cum_f += f
    cum_gradf += gradf.T

