#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat

#===============================================================

dpath = '/home/mtageld/Desktop/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/GBMLGG_Integ_Preprocessed.mat'

data = loadmat(dpath)

N = len(data['Survival'][0])
T = data['Survival'].reshape([N,])
C = data['Censored'].reshape([N,])

# PLAYING AROUND ********
T = T[0:30]
C = C[0:30]


#==============================================================

# find unique times
t = np.unique(T[C == 0])

# initialize count vectors
f = np.zeros(t.shape)
d = np.zeros(t.shape)
n = np.zeros(t.shape)

# generate counts
for i in range(len(t)):
    n[i] = np.sum(T >= t[i]) # <- no at risk
    d[i] = np.sum(T[C == 0] == t[i]) # <- no of events

#
# Use K-M estimator    
#

# calculate probabilities
f = (n - d) / n
f = np.cumprod(f)

# append beginning and end points
t_start = np.array([0])
f_start = np.array([1])
t_end = np.array([T.max()])
f_end = np.array([f[-1]])
t = np.concatenate((t_start, t, t_end), axis=0)
f = np.concatenate((f_start, f, f_end), axis=0)

# Get an estimate of the KM survivor function
#==============================================================

#t_unit = 30 # months

def discretize_km(t, f, t_unit = 30):
    
    """ discretize kaplan meier to denoise"""

    timepoints = np.arange(0, t[-1], t_unit)
    freqs = np.zeros(timepoints.shape)
    
    for tidx, tpoint in enumerate(timepoints):
        freqs[tidx] = f[np.where(t > tpoint)[0][0]-1]
    
    return timepoints, freqs

# PLAYING AROUND!! ****
t0 = t.copy()
f0 = f.copy()

t30, f30 = discretize_km(t0, f0, 30)
t180, f180 = discretize_km(t0, f0, 180)
t365, f365 = discretize_km(t0, f0, 365)
