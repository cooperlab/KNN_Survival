#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat
#from scipy.optimize import curve_fit

#===============================================================

dpath = '/home/mtageld/Desktop/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/GBMLGG_Integ_Preprocessed.mat'

data = loadmat(dpath)

T = data['Survival']
C = data['Censored']

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
#t_end = np.array([T.max()])

# Get estimate of K-M survivor function
#t = np.concatenate((t_start, t, t_end), axis=0)
t0 = np.concatenate((t_start, t), axis=0)
f0 = np.concatenate((f_start, f), axis=0)

#==============================================================

#def f_exp(t, lamb):
#    """An exponential survival function"""
#    return (np.exp(-lamb * t))
#
#
#lamb, _ = curve_fit(f_exp, t, f)

from scipy.interpolate import splprep, splev
tck, u = splprep([t0, f0], s= 0)
new_points = splev(u, tck)
t1 = np.array(new_points[0])
f1 = np.array(new_points[1])


