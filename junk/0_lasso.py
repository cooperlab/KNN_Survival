# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 15:19:09 2017

@author: mohamed
"""

import tensorflow as tf

#%%============================================================================
# Elastic net penalty
#==============================================================================

#
# Inspired by: 
# https://github.com/glm-tools/pyglmnet/blob/master/pyglmnet/pyglmnet.py
#

def _penalty(W, alpha, lambd, group):
    
    """
    Elastic net penalty. Inspired by: 
    https://github.com/glm-tools/pyglmnet/blob/master/pyglmnet/pyglmnet.py
    """
    
    with tf.name_scope("Elastic_net_penalty"):
        # Lasso-like penalty
        L1penalty = lambd * tf.reduce_sum(tf.abs(W), axis=0)
        
        # Compute the L2 penalty (ridge-like)
        L2penalty = lambd * tf.reduce_sum(W ** 2, axis=0)
            
        
        # Combine L1 and L2 penalty terms
        P = 0.5 * (alpha * L1penalty + (1 - alpha) * L2penalty)
    
    return P