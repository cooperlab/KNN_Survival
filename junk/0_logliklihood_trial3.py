# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 23:05:39 2017

@author: mohamed
"""

import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')
import SurvivalUtils as sUtils

import tensorflow as tf
import numpy as np

#%% 
#
# Generate simulated data
#
n = 30; d = 140
X_input = np.random.rand(n, d)
T = np.random.randint(0, 300, [n,])
C = np.random.randint(0, 2, [n,])
T, O, at_risk, X_input = sUtils.calc_at_risk(T, 1-C, X_input)

Pij = np.random.rand(n, n)
Pij = Pij / np.sum(Pij, axis=1)

#%%

# Get at-risk mask (to be multiplied by Pij)
Pij_mask = np.zeros((n, n))
for idx in range(n):
    # only observed cases
    if O[idx] == 1:
        # only at-risk cases
        Pij_mask[idx, at_risk[idx]:] = 1

#%%

def numpy_version_of_cost(Pij, Pij_mask):
    
    """
    Numpy version of the negative log likelihood. 
    
    IMPORTANT NOTE:
    Keep in mind that patients are assumed to be sorted in 
    ascending order of their time-to-event
    """
    
    # Restrict Pij to observed and at-risk cases
    Pij = Pij * Pij_mask
    
    return np.sum(Pij)

#%%

def tensorflow_version_of_cost(Pij, Pij_mask):
    
    """
    tensorflow version of the negative log likelihood. 
    
    IMPORTANT NOTE:
    Keep in mind that patients are assumed to be sorted in 
    ascending order of their time-to-event
    """
    
    # -----------------------------
    # Add to graph (for demo)
    #tf.reset_default_graph()
    Pij = tf.Variable(Pij)
    Pij_mask = tf.Variable(Pij_mask)
    # -----------------------------
    
    # Restrict Pij to observed and at-risk cases
    Pij = tf.multiply(Pij, Pij_mask)
    
    # cost the sum of Pij of at-risk cases over
    # all observed cases
    cost = tf.reduce_sum(Pij)
    
    # run session to fetch cost
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())    
        c = cost.eval()
    
    return c
    
#%%

# check that the numpy and tensforlow results are the same
c_np = tensorflow_version_of_cost(output, O, at_risk)
c_tf = numpy_version_of_cost(output, O, at_risk)
