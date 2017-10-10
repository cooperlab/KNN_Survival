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

# predicted ** risk **
# Note that predicted risk HAS to be > 0 so that when we take
# the log we don't end up with a numerical error
output = np.random.rand(n)

#%%

def numpy_version_of_cost(output, O, at_risk):
    
    """
    Numpy version of the negative log likelihood. 
    
    IMPORTANT NOTE:
    Keep in mind that patients are assumed to be sorted in 
    ascending order of their time-to-event
    """
    
    def exclusive_cumSum(x):
        """
        find cumulative sum of all PREVIOUS indices (excluding current one).
        Args:
            x - np array
        Returns:
            x - same size as x
        """
        x = np.cumsum(x)[:-1]
        return np.concatenate(([0], x))
        
    # exponentiate predicted risk and flip so that at-risk cases appear before 
    # each patient. i.e. those with longer survival appear earlier
    exp = np.flip(np.exp(output), axis=0)
    
    # Take the cumulative sum -> now each index i contains the sum of predictions 
    # for all at-risk cases (i.e. those appearing before i).
    # ALERT: this causes a problem (for now): Cases that have the exact same 
    #        survival time will (for now) have different partial sum values 
    #        because they appear consecutively
    partial_sum_a = exclusive_cumSum(exp)
    
    # flip so that cases are in the same order they
    # were before. Note that 1 is added so that the 
    # zeroth case (with a cumSum of zero) does not result
    # in a numerical error when the log of predictions is done later
    partial_sum = np.flip(partial_sum_a, axis=0) + 1
    
    # Now make sure cases that have the exact same survival also get the 
    # exact same risk partial sum values
    partial_sum = partial_sum[at_risk]
    
    # Get log likelihood difference
    log_at_risk = np.log(partial_sum + 1e-50)
    diff = output - log_at_risk
    
    # Restrict to only observed cases
    times = diff * O
    
    # cost is negative log likelihood
    cost = - np.sum(times)
    
    return cost

#%%

def tensorflow_version_of_cost(output, O, at_risk):
    
    """
    tensorflow version of the negative log likelihood. 
    
    IMPORTANT NOTE:
    Keep in mind that patients are assumed to be sorted in 
    ascending order of their time-to-event
    """
    
    # -----------------------------
    # Add to graph (for demo)
    #tf.reset_default_graph()
    output = tf.Variable(output)
    O = tf.Variable(O, dtype='float64')
    at_risk = tf.Variable(at_risk)
    # -----------------------------
    
    # exponentiate predicted risk and flip so that at-risk cases appear before 
    # each patient. i.e. those with longer survival appear earlier
    exp = tf.reverse(tf.exp(output), axis=[0])
    
    # Take the cumulative sum -> now each index i contains the sum of predictions 
    # for all at-risk cases (i.e. those appearing before i).
    # ALERT: this causes a problem (for now): Cases that have the exact same 
    #        survival time will (for now) have different partial sum values 
    #        because they appear consecutively
    partial_sum_a = tf.cumsum(exp, exclusive=True)
    
    # flip so that cases are in the same order they
    # were before. Note that 1 is added so that the 
    # zeroth case (with a cumSum of zero) does not result
    # in a numerical error when the log of predictions is done later
    partial_sum = tf.reverse(partial_sum_a, axis=[0]) + 1
    
    # Now make sure cases that have the exact same survival also get the 
    # exact same risk partial sum values
    partial_sum = tf.gather(partial_sum, at_risk)
    
    # Get log likelihood difference
    log_at_risk = tf.log(partial_sum + 1e-50)
    diff = output - log_at_risk
    
    # Restrict to only observed cases
    times = diff * O
    
    # cost is negative log likelihood
    cost = - tf.reduce_sum(times)
    
    # run session to fetch cost
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())    
        c = cost.eval()
    
    return c
    
#%%

# check that the numpy and tensforlow results are the same
c_np = tensorflow_version_of_cost(output, O, at_risk)
c_tf = numpy_version_of_cost(output, O, at_risk)
