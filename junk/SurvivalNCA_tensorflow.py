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
import SurvivalUtils as sUtils
import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.INFO)


#%%============================================================================
# ---- J U N K ----------------------------------------------------------------
#==============================================================================

from scipy.io import loadmat

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

# Generate survival status - discretized into months
aliveStatus = sUtils.getAliveStatus(Survival, Censored, scale = 30)


#%%============================================================================
# --- P R O T O T Y P E S -----------------------------------------------------
#==============================================================================

LEARN_RATE = 1e-5



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

# Get dims
N, D = np.int32(data.shape)
T = np.int32(aliveStatus.shape[1]) # no of time points

# Graph input
X = tf.placeholder("float32", [N, D], name='X')
alive = tf.placeholder("int32", [N, T], name='alive')

# Get mask of available survival status at different time points
avail_mask = tf.cast((aliveStatus >= 0), tf.int32, name='avail_mask')

# Initialize A to a scaling matrix
A = np.zeros((D, D))
epsilon = 1e-7 # to  avoid division by zero
np.fill_diagonal(A, 1./(data.max(axis=0) - data.min(axis=0) + epsilon))
A = np.float32(A)
A = tf.Variable(A, name='A')

# initilize A and transform input
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
    t = tf.Print(t, [t, i], message='t, i = ')

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
cumSum = tf.cast(tf.Variable([0.0]), tf.float32)
t = tf.cast(tf.constant(0), tf.int32)
i = tf.cast(tf.constant(0), tf.int32)

def t_not_max(t, i, cumSum):
    return tf.less(t, 5) #T) # DEBUG!!!

# loop through time points
_, _, cumSum = tf.while_loop(t_not_max, add_to_cumSum, [t, i, cumSum])

cost = -cumSum

optimizer = tf.train.AdamOptimizer(LEARN_RATE).minimize(cost)


#%%============================================================================
# Launch graph
#==============================================================================


with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    feed = {X: data, alive: aliveStatus}
    
    # fetches = [t, i, ignoreMask, softmax, match, Pi, cumSum]
    # fetch_names = ['t', 'i', 'ignoreMask', 'softmax', 'match', 'Pi', 'cumSum']
    
    fetches = [optimizer]
    fetch_names = ['optimizer']
    
    f = sess.run(fetches, feed_dict = feed)
    
    fetched = {}
    for i,j in enumerate(f):
        fetched[fetch_names[i]] = j

    
