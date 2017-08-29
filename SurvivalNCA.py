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
#import scipy.optimize as opt

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

DIMS = data.shape[1] # set DIMS < D to reduce dimensions

OBJECTIVE = 'Mahalanobis'
THRESH = None # None or between [0, 1], the smaller the more emphasis on closer neighbors
LEARN_RATE = 0.01
MONITOR_STEP = 1

# no of patients chosen randomly each time point
N_SUBSET = 30

#%%============================================================================
# Setting things up
#==============================================================================

# Get dims
N, D = data.shape
T = aliveStatus.shape[1] # no of time points
epsilon = 1e-7 # to  avoid division by zero

# Initialize A to a scaling matrix
#A = np.eye(D, DIMS)
A = np.zeros((D, DIMS))
np.fill_diagonal(A, 1./(data.max(axis=0) - data.min(axis=0) + epsilon))

#A = (1./(data.max(axis=0) - data.min(axis=0) + epsilon)).reshape(D, 1)
#A = np.ones([D, 1])

#A_init = A.copy()


#%%============================================================================
# Define objective function
#==============================================================================

def survival_nca_cost(A, data, aliveStatus, 
                      OBJECTIVE= 'Mahalanobis', THRESH = None,
                      N_SUBSET = 20):
    
    """Gets cumulative cost and gradient for all time points"""

    # initialize cum_f and cum_gradf
    cum_f = 0
    cum_gradf = np.zeros(A.shape)
    
    #t = 20
    for t in range(T):
    
        print("t = {} of {}".format(t, T-1))
            
        # Get patients with known survival status at time t
        Y = aliveStatus[:, t]
        keep = Y >= 0 
        # proceed only if there's enough known patients
        if np.sum(0 + keep) < N_SUBSET:
            print("skipping current t ...")
            continue 
        Y = Y[keep]
        X = data[keep, :]
        
        # keep a random subset of patients (for efficiency)
        keep = np.random.randint(0, X.shape[0], N_SUBSET)
        Y = Y[keep]
        X = X[keep, :]
        
        if OBJECTIVE == 'Mahalanobis':
            f, gradf = nca_cost.cost(A.T, X.T, Y, threshold=THRESH)
        elif OBJECTIVE == 'KL-divergence':
            f, gradf = nca_cost.cost_g(A.T, X.T, Y, threshold=THRESH)
            
        cum_f += f
        cum_gradf += gradf.T # sum of derivative is derivative of sum

    return [cum_f, cum_gradf]


#%%============================================================================
# Optimize objective function
#==============================================================================

# This does not work - quickly runs out of memory
#res = opt.minimize(fun = survival_nca_cost,
#                   x0 = A,
#                   args = (data, aliveStatus, OBJECTIVE, THRESH),
#                   jac = True,
#                   method = 'BFGS')


costs = []
step = 0
 
try: 
    while True:
        
        print("\n--------------------------------------------")
        print("---- STEP = " + str(step))
        print("--------------------------------------------\n")
        
        
        [cum_f, cum_gradf] = survival_nca_cost(A, data, aliveStatus, 
                                               OBJECTIVE = OBJECTIVE, 
                                               THRESH = THRESH, 
                                               N_SUBSET = N_SUBSET)
        
        # update A
        A -= LEARN_RATE * cum_gradf
        
        # update costs
        costs.append([step, cum_f])
        
        # monitor
        if step % MONITOR_STEP == 0:
            
            cs = np.array(costs)
            
            sUtils.plotMonitor(arr= cs, title= "cost vs. epoch", 
                               xlab= "epoch", ylab= "cost", 
                               savename= RESULTPATH + "cost.svg")
        
        step += 1
        
except KeyboardInterrupt:
    pass

#%%============================================================================
# Now parse the learned matrix A and save
#==============================================================================

def getRanks(A):
    
    Ax = np.dot(data, A)

    fvars = np.std(Ax, 0).reshape(DIMS, 1)
    fidx = np.arange(len(A)).reshape(DIMS, 1)
    
    fvars = np.concatenate((fidx, fvars), 1)
    fvars = fvars[fvars[:,1].argsort()][::-1]
    keep = fvars[:,1] < 100000
    fvars = fvars[keep,:]
    
    fnames_ranked = fnames[np.int32(fvars[:,0])]
    
    return fnames_ranked

ranks_init = getRanks(np.eye(D, DIMS))
ranks = getRanks(A)

# Save analysis result
result = {'A': A,
          'ranks_init': ranks_init,
          'ranks': ranks,
          'LEARN_RATE': LEARN_RATE,}

savemat(RESULTPATH + 'result', result)


#%%



