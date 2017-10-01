# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 16:33:24 2017

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

#%%

# -----------------------------
# Add to graph (for demo)
tf.reset_default_graph()
X_input = tf.Variable(X_input)
O = tf.Variable(O, dtype='float64')
at_risk = tf.Variable(at_risk)

# for now, let's assume we already NCA_transformed X
X_transformed = X_input

# no of feats and split size
dim_input = 140
per_split_feats = 30
#----------------------------
#%%
n_splits = int(dim_input / per_split_feats)
n_divisible = n_splits * per_split_feats
X_split = tf.split(X_transformed[:,0:n_divisible], n_splits, axis=1)
X_split.append(X_transformed[:,n_divisible:])


# get norm along first feature set
normAX = X_split[0][None, :, :] - X_split[0][:, None, :]
normAX = tf.reduce_sum(normAX ** 2, axis=2)
 
for split in range(1, len(X_split)): 
 
    # Expand dims of AX to [n_samples, n_samples, n_features], where
    # each "channel" in the third dimension is the difference between
    # one sample and all other samples along one feature
    norm_thisFeatureSet = X_split[split][None, :, :] - \
                          X_split[split][:, None, :]
    
    norm_thisFeatureSet = tf.reduce_sum(norm_thisFeatureSet ** 2, axis=2)
    
    # add to existing cumulative sum    
    normAX = normAX + norm_thisFeatureSet
    
# Calculate Pij, the probability that j will be chosen 
# as i's neighbor, for all i's. Pij has shape
# [n_samples, n_samples] and ** is NOT symmetrical **.
# Because the data is normalized using softmax, values
# add to 1 in rows, that is i (central patients) are
# represented in rows
denomSum = tf.reduce_sum(tf.exp(-normAX), axis=0)
epsilon = 1e-50
denomSum = denomSum + epsilon            

Pij = tf.exp(-normAX) / denomSum[:, None]
