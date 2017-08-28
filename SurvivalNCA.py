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

from scipy.io import loadmat, savemat
import numpy as np
import SurvivalUtils as sUtils
import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.INFO)


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

# Get split indices
#splitIdxs = sUtils.getSplitIdxs(data)

n = 100
data = data[0:n,:]
Survival = Survival[0:n,:]
Censored = Censored[0:n,:]

# Generate survival status - discretized into months
aliveStatus = sUtils.getAliveStatus(Survival, Censored, scale = 30)


#%%============================================================================
# --- P R O T O T Y P E S -----------------------------------------------------
#==============================================================================

RESULTPATH = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/tmp/"

LEARN_RATE = 50
D_new = data.shape[1] # set D_new < D to reduce dimensions
MONITOR_STEP = 10


#%%============================================================================
# Setting things up
#==============================================================================

# Get dims
N, D = np.int32(data.shape)
T = np.int32(aliveStatus.shape[1]) # no of time points

# Initialize A to a scaling matrix
A_init = np.zeros((D, D_new))
epsilon = 1e-7 # to  avoid division by zero
np.fill_diagonal(A_init, 1./(data.max(axis=0) - data.min(axis=0) + epsilon))
A_init = np.float32(A_init)


#%%============================================================================
# Building the compuational graph
#==============================================================================

#
# Some parts were Modified from: 
# https://all-umass.github.io/metric-learn/_modules/metric_learn/nca.html#NCA
#

# -----------------------------------------------------------------------------
# Basic ground work
# -----------------------------------------------------------------------------

tf.reset_default_graph()

# Graph input
X = tf.placeholder("float32", [N, D], name='X')
alive = tf.placeholder("int32", [N, T], name='alive')
A = tf.Variable(A_init, name='A')

# Get mask of available survival status at different time points
avail_mask = tf.cast((aliveStatus >= 0), tf.int32, name='avail_mask')

# Transform input
AX = tf.matmul(X, A)  # shape (N, D)


# -----------------------------------------------------------------------------
# Define core functions
# -----------------------------------------------------------------------------

def add_to_cumSum(t, i, cumSum):
    
    """ 
    Gets "probability" of patient i's survival status being correctly 
    predicted at time t and adds it to running cumSum for all points 
    whose survival status is known at time t ... this only happens
    if survival status of i is known at time t 
    """
    
    # Monitor progress
    # Note:  This is not currently compatible with jupyter notebook
    # t = tf.Print(t, [t, i], message='t, i = ')

    # Get ignore mask ->  give unknown status zero weight
    # Note that the central point itself is not ignored because 
    # tensorflow only allows item assignment for variables (not tensores)
    # and it does not allow the creation of variables inside while loops
    ignoreMask = avail_mask[:, t] # unavailable labels at time t are -> 0
    ignoreMask = tf.cast(ignoreMask, tf.float32)
    
    # Calculate normalized feature similarity metric between central point 
    # and those cases with available survival status at time t
    softmax = tf.exp(-tf.reduce_sum((AX[i,:] - AX)**2, axis=1)) # shape (n)
    softmax = tf.multiply(softmax, ignoreMask)
    softmax = softmax / tf.reduce_sum(softmax)
    
    # Get label match mask (i.e. i is alive and j is alive, same if i is dead)
    # note there is no need to set central/censored points to zero since this will
    # be multiplied by softmax which already has zeros at these locations
    match = tf.cast(tf.equal(alive[:, t], alive[i, t]), tf.float32)
    
    # Get "probability" of correctly classifying i at time t
    Pi = tf.reduce_sum(tf.multiply(softmax, match))
    
    # add to cumSum - only if survival status of i is known at time t 
    survival_is_known = tf.cast(tf.equal(avail_mask[i, t], 1), tf.float32)
    cumSum = cumSum + (Pi * survival_is_known)
    
    # increment t if last patient
    t = tf.cond(tf.equal(i, N-1), lambda: tf.add(t, 1), lambda: tf.add(t, 0))

    # increment i (or reset it if last patient)
    i = tf.cond(tf.equal(i, N-1), lambda: tf.multiply(i, 0), lambda: tf.add(i, 1))
    
    return t, i, cumSum

# -----------------------------------------------------------------------------
# Now go through all time points and patients
# -----------------------------------------------------------------------------

# initialize
cumSum = tf.cast(tf.Variable([0.0], name='cumSum'), tf.float32)
t = tf.cast(tf.constant(0, name='t'), tf.int32)
i = tf.cast(tf.constant(0, name='i'), tf.int32)

def t_not_max(t, i, cumSum):
    return tf.less(t, T)

# loop through time points
_, _, cumSum = tf.while_loop(t_not_max, add_to_cumSum, [t, i, cumSum])

cost = -cumSum
optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(cost)

## Calculate gradient
#d = tf.gradients(cumSum, A, name='d')
## Update A
#A_new = A + (tf.reshape(tf.multiply(d, A), [D, D_new]) * LEARN_RATE)


#%%============================================================================
# Launch session
#==============================================================================

with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    cumsums = []
    step = 0
    
    diffs = []
     
    try: 
        while True:
            
            print("\n--------------------------------------------")
            print("---- STEP = " + str(step))
            print("--------------------------------------------\n")
            
            fetches = [optimizer, A, cumSum]
            feed = {X: data, alive: aliveStatus}
            
            _, A_current, cumSum_current = sess.run(fetches, feed_dict = feed)
            
            # initialize cumsums and total abs change in matrix A
            cumsums.append([step, cumSum_current[0]])
            diffs.append([step, np.sum(np.abs(A_current - A_init))])
            
            # monitor
            if step % MONITOR_STEP == 0:
                
                cs = np.array(cumsums)                
                df = np.array(diffs)
                
                sUtils.plotMonitor(arr= cs, title= "cumSum vs. epoch", 
                                   xlab= "epoch", ylab= "cumSum", 
                                   savename= RESULTPATH + "cumSums.svg")
                                   
                
                sUtils.plotMonitor(arr= df, title= "deltaA vs. epoch", 
                                   xlab= "epoch", ylab= "deltaA", 
                                   savename= RESULTPATH + "deltaA.svg")
            
            step += 1
            
    except KeyboardInterrupt:
        pass



#%%============================================================================
# Now parse the learned matrix A and save
#==============================================================================

def getRanks(A):
    w = np.diag(A).reshape(D_new, 1)
    fidx = np.arange(len(A)).reshape(D_new, 1)
    
    w = np.concatenate((fidx, w), 1)
    w = w[w[:,1].argsort()][::-1]
    tokeep = w[:,1] < 100000
    w = w[tokeep,:]
    
    
    fnames = Data['Integ_Symbs']
    fnames = fnames[np.int32(w[:,0])]
    
    return fnames

ranks_init = getRanks(A_init)
ranks_current = getRanks(A_current)

# Save analysis result
result = {'A_init': A_init,
          'A_current': A_current,
          'ranks_init': ranks_init,
          'ranks_current': ranks_current,
          'LEARN_RATE': LEARN_RATE,}

savemat(RESULTPATH + 'result', result)
