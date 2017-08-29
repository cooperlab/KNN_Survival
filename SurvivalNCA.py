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
#dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Gene.mat"
#dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat"

Data = loadmat(dpath)

data = np.float32(Data['Integ_X'])
#data = np.float32(Data['Gene_X'])

if np.min(Data['Survival']) < 0:
    Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1

Survival = np.int32(Data['Survival'])
Censored = np.int32(Data['Censored'])
fnames = Data['Integ_Symbs']
#fnames = Data['Gene_Symbs']

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

self.RESULTPATH = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/tmp/"
self.description = "nca_GBMLGG_"

self.DIMS = data.shape[1] # set DIMS < D to reduce dimensions

self.OBJECTIVE = 'Mahalanobis'
self.THRESH = None # None or between [0, 1], the smaller the more emphasis on closer neighbors
self.LEARN_RATE = 0.01
self.MONITOR_STEP = 1

# no of patients chosen randomly each time point
self.N_SUBSET = 10

#%%============================================================================
# Setting things up
#==============================================================================

# Get dims
self.N, self.D = self.data.shape
self.T = aliveStatus.shape[1] # no of time points
epsilon = 1e-7 # to  avoid division by zero

# Initialize A to a scaling matrix
#A = np.eye(D, DIMS)
self.A = np.zeros((D, DIMS))
np.fill_diagonal(self.A, 
                 1./(self.data.max(axis=0) - self.data.min(axis=0) + epsilon))



#%%============================================================================
# Core model
#==============================================================================

def _survival_nca_cost(self):
    
    """Gets cumulative cost and gradient for all time points"""

    # initialize cum_f and cum_gradf
    cum_f = 0
    cum_gradf = np.zeros(self.A.shape)
    
    #t = 20
    for t in range(self.T):
    
        print("t = {} of {}".format(t, self.T-1))
            
        # Get patients with known survival status at time t
        Y = self.aliveStatus[:, t]
        keep = Y >= 0 
        # proceed only if there's enough known patients
        if np.sum(0 + keep) < self.N_SUBSET:
            print("skipping current t ...")
            continue 
        Y = Y[keep]
        X = self.data[keep, :]
        
        # keep a random subset of patients (for efficiency)
        keep = np.random.randint(0, X.shape[0], self.N_SUBSET)
        Y = Y[keep]
        X = X[keep, :]
        
        if self.OBJECTIVE == 'Mahalanobis':
            f, gradf = nca_cost.cost(self.A.T, X.T, Y, threshold=self.THRESH)
        elif self.OBJECTIVE == 'KL-divergence':
            f, gradf = nca_cost.cost_g(self.A.T, X.T, Y, threshold=self.THRESH)
            
        cum_f += f
        cum_gradf += gradf.T # sum of derivative is derivative of sum

    return [cum_f, cum_gradf]


#==========================================================================

def train(self):
    
    """ learns feature matrix A to minimize objective function"""
    
    self.costs = []
    step = 0
     
    try: 
        while True:
            
            print("\n--------------------------------------------")
            print("---- STEP = " + str(step))
            print("--------------------------------------------\n")
            
            [cum_f, cum_gradf] = self._survival_nca_cost(self.A)
            # update A
            self.A -= self.LEARN_RATE * cum_gradf
            
            # Discard non-diagnoal terms - 
            # Uncomment if you just want linear scaling of features,
            # but expect worse performance without much gain in 
            # interpretability. (columns of Ax still correspond to 
            # columns of x without this step)
            # A *= np.eye(A.shape[0], A.shape[1])
    
            # update costs
            self.costs.append([step, cum_f])
            
            # monitor
            if (step % self.MONITOR_STEP == 0) and (step > 0):
                cs = np.array(self.costs)
                self._plotMonitor(arr= cs, title= "cost vs. epoch", 
                                  xlab= "epoch", ylab= "cost", 
                                  savename= self.RESULTPATH + 
                                  self.description + "cost.svg")
            
            step += 1
            
    except KeyboardInterrupt:
        print("\n Finished training model.")


#==========================================================================

def rankFeats(self):
    
    """rank features by how variant they are after the new transformation"""

    def _getRanks(A):
        
        Ax = np.dot(self.data, A)
    
        fvars = np.std(Ax, 0).reshape(self.DIMS, 1)
        fidx = np.arange(len(A)).reshape(self.DIMS, 1)
        
        fvars = np.concatenate((fidx, fvars), 1)
        fvars = fvars[fvars[:,1].argsort()][::-1]
        
        fnames_ranked = fnames[np.int32(fvars[:,0])]
        
        return fvars, fnames_ranked
    
    self.fvars_init, self.ranks_init = _getRanks(np.eye(self.D, self.DIMS))
    self.fvars, self.ranks = _getRanks(self.A)


#%%===========================================================================
# Helper methods
#==============================================================================

# The following load/save methods are inspired by:
    # https://stackoverflow.com/questions/2345151/
    # how-to-save-read-class-wholly-in-python
    
def save(self):
    
    """save class as ModelAttributes.txt"""
    
    print("Saving model attributes ...")
    self._updateStepCount()
    with open(self.RESULTPATH + self.description + 'ModelAttributes.txt','wb') as file:
        file.write(_pickle.dumps(self.__dict__))
        file.close()

#==========================================================================

def load(self, LOADPATH):
    
    """try to load ModelAttributes.txt"""
    
    print("Loading model attributes ...")
    with open(LOADPATH,'rb') as file:
        dataPickle = file.read()
        file.close()
        self.__dict__ = _pickle.loads(dataPickle)
        
#==========================================================================

def _plotMonitor(self, arr, title, xlab, ylab, savename):
                    
    """ plots cost/other metric to monitor progress """
    
    print("Plotting " + title)
    
    fig, ax = plt.subplots() 
    ax.plot(arr[:,0], arr[:,1], 'b', linewidth=1.5, aa=False)
    plt.title(title, fontsize =16, fontweight ='bold')
    plt.xlabel(xlab)
    plt.ylabel(ylab) 
    plt.tight_layout()
    plt.savefig(savename)
    plt.close() 


#%%
#%%
#%%
#%%
result = 

# Save analysis result
result = {'A': self.A,
          'ranks_init': self.ranks_init,
          'ranks': self.ranks}

savemat(self.RESULTPATH + self.description + 'result', result)


#%%============================================================================
# Visualize results
#==============================================================================

from matplotlib import cm
import matplotlib.pylab as plt

Ax = np.dot(data, A)
fidx = np.arange(len(A)).reshape(DIMS, 1)

#%%
# plot stdev change
#

fig, ax = plt.subplots()

ax.plot(fidx, fvars[:,1], 'b', linewidth=1.5, aa=False)
ax.plot(fidx, fvars_init[:,1], 'k--', linewidth=1.5, aa=False)

plt.ylim(ymax = 1.5)

plt.title("feature stdev after transformation", fontsize =16, fontweight ='bold')
plt.xlabel("feature index")
plt.ylabel("feature stdev - \nbefore (k--) and after (b-) transformation")
plt.savefig(RESULTPATH + "fvars.svg")
plt.close()

#%%
# scatter patients by top two features
#

fig, ax = plt.subplots()

keep = (Censored == 0).reshape(N)
X1 = Ax[keep, int(fvars[0,0])]
X2 = Ax[keep, int(fvars[1,0])]
Ys = Survival[keep,:]

colors = cm.seismic(np.linspace(0, 1, len(Ys)))
#cs = [colors[i//len(X1)] for i in range(len(Ys)*len(X1))] #could be done with numpy's repmat

ax.scatter(X1, X2, color=colors)
plt.title("Top features (transformed) vs survival (color)", 
          fontsize =16, fontweight ='bold')
plt.xlabel(str(ranks[0]), fontsize=5)
plt.ylabel(ranks[1], fontsize=5)
plt.savefig(RESULTPATH + "scatterByTop2.svg")
plt.close()
