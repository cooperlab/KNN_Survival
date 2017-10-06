#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 12:57:55 2017

@author: mohamed
"""

#import os
#import sys
#def conditionalAppend(Dir):
#    """ Append dir to sys path"""
#    if Dir not in sys.path:
#        sys.path.append(Dir)
#cwd = os.getcwd()
#conditionalAppend(cwd)

import tensorflow as tf

#import ProjectUtils as pUtils

#%%============================================================================
# Computational graph class
#==============================================================================

class comput_graph(object):
    
    """
    Builds the computational graph for Survival NCA.
    """
    
    def __init__(self, dim_input, 
                 ALPHA = 0.5,
                 LAMBDA = 1.0,
                 SIGMA = 1.0,
                 OPTIM = 'GD',
                 LEARN_RATE = 0.01,
                 per_split_feats = 500,
                 ROTATE = False):
        
        """
        Instantiate a computational graph for survival NCA.
        
        Args:
        ------
        dim_input - no of features
        ALPHA - weighing of L1 penalty (vs L2)
        LAMBDA - weighting of the values of the penalties
        SIGMA - controls emphasis on nearest neighbors 
        OPTIM - type of optimizer
        LEARN_RATE - learning rate
        per_split_feats - if this number is smaller than the total
                          no of features, it controld how many features at
                          a time to consider whan calculating Pij. The smaller
                          the more likely the matrix is to fit into memory, 
                          with no effect on the end result
        ROTATE - when this is true, A is not limited to a scaling matrix
        """
        
        #print("Building computational graph for survival NCA.")
        #pUtils.Log_and_print("Building computational graph for survival NCA.")    
        
        # set up instace attributes
        self.dim_input = dim_input
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.SIGMA = SIGMA
        self.OPTIM = OPTIM
        self.LEARN_RATE = LEARN_RATE
        self.per_split_feats = per_split_feats
        self.ROTATE = ROTATE
        
        # clear lurking tensors
        tf.reset_default_graph()
        
        #pUtils.Log_and_print("Adding placeholders.")
        self.add_placeholders()
        
        #pUtils.Log_and_print("Adding linear feature transform.")
        self.add_linear_transform()
            
        #pUtils.Log_and_print("Adding regularized weighted log likelihood.")
        self.add_cost()
        
        #pUtils.Log_and_print("Adding optimizer.")
        self.add_optimizer()
        
        #pUtils.Log_and_print("Finished building graph.")


    #%%========================================================================
    # Add placeholders to graph  
    #==========================================================================
    
    def add_placeholders(self):
    
        """ Adds graph inputs as placeholders in graph """
        
        with tf.variable_scope("Inputs"):
        
            self.X_input = tf.placeholder("float", [None, self.dim_input], name='X_input')
            
            self.T = tf.placeholder("float", [None], name='T')
            self.O = tf.placeholder("float", [None], name='O')
            self.At_Risk = tf.placeholder("float", [None], name='At_Risk')
            
            # type conversions
            self.T = tf.cast(self.T, tf.float32)
            self.O = tf.cast(self.O, tf.int32)
            self.At_Risk = tf.cast(self.At_Risk, tf.int32)
            
                 
    #%%========================================================================
    # Linear transformation (for interpretability)
    #==========================================================================

    def add_linear_transform(self):
        
        """ 
        Transform features in a linear fashion for better interpretability
        """
        
        with tf.variable_scope("linear_transform"):
            
            
            if self.ROTATE:
                self.W = tf.get_variable("weights", shape=[self.dim_input, self.dim_input], 
                                initializer= tf.contrib.layers.xavier_initializer())
            else:
                # feature scales/weights
                self.w = tf.get_variable("weights", shape=[self.dim_input], 
                                initializer= tf.contrib.layers.xavier_initializer())
                # diagonalize and matmul
                self.W = tf.diag(self.w)
            
            self.X_transformed = tf.matmul(self.X_input, self.W)
    
    #%%========================================================================
    # Get Pij 
    #==========================================================================

    def _get_Pij(self):
        
        """ 
        Calculate Pij, the probability that j will be chosen 
        as i's neighbor, for all i's
        Inspired by: https://github.com/RolT/NCA-python
        """        
        
        with tf.name_scope("getting_Pij"):
            
            if self.per_split_feats >= self.dim_input:

                # find distance along all features in one go
                normAX = self.X_transformed[None, :, :] - \
                         self.X_transformed[:, None, :]
                normAX = tf.reduce_sum(normAX ** 2, axis=2)
            
            else:
                # find distance along batches of features and cumsum
                n_splits = int(self.dim_input / self.per_split_feats)
                n_divisible = n_splits * self.per_split_feats
                X_split = tf.split(self.X_transformed[:,0:n_divisible], n_splits, axis=1)
                X_split.append(self.X_transformed[:,n_divisible:])
    
                # get norm along first feature set
                print("\tsplit 0")
                normAX = X_split[0][None, :, :] - X_split[0][:, None, :]
                normAX = tf.reduce_sum(normAX ** 2, axis=2)
                 
                for split in range(1, len(X_split)): 
                    
                    print("\tsplit {}".format(split))
                 
                    # Expand dims of AX to [n_samples, n_samples, n_features], where
                    # each "channel" in the third dimension is the difference between
                    # one sample and all other samples along one feature
                    norm_thisFeatureSet = X_split[split][None, :, :] - \
                                          X_split[split][:, None, :]
                    
                    norm_thisFeatureSet = tf.reduce_sum(norm_thisFeatureSet ** 2, axis=2)
                    
                    # add to existing cumulative sum    
                    normAX = normAX + norm_thisFeatureSet
                
            # Calculate Pij, the probability that j will be chosen 
            # as i's closest neighbor, for all i's. Pij has shape
            # [n_samples, n_samples] and ** is NOT symmetrical **.
            # Because the data is normalized using softmax, values
            # add to 1 in rows. That is, i (central patients) are
            # represented in rows
            
            def kernel_function(z):
                return tf.exp(-z * self.SIGMA)

            denomSum = tf.reduce_sum(kernel_function(normAX), axis=0)
            epsilon = 1e-50
            denomSum = denomSum + epsilon            
            
            self.Pij = kernel_function(normAX) / denomSum[:, None]
    

    #%%========================================================================
    #  Loss function - weighted log likelihood   
    #==========================================================================

    def add_cost(self):
        
        """
        Adds penalized weighted likelihood to computational graph        
        """
    
        # Get Pij, probability j will be i's neighbor
        self._get_Pij()
        
        def _add_to_cumSum(Idx, cumsum):
        
            """Add patient to log partial likelihood sum """
            
            # Get Pij of at-risk cases from this patient's perspective
            Pij_thisPatient = self.Pij[Idx, self.At_Risk[Idx]:tf.size(self.T)-1]
            
            # Get sum
            Pij_thisPatient = tf.reduce_sum(Pij_thisPatient)
            cumsum = tf.add(cumsum, Pij_thisPatient)
            
            return cumsum
    
        def _add_if_observed(Idx, cumsum):
        
            """ Add to cumsum if current patient'd death time is observed """
            
            with tf.name_scope("add_if_observed"):
                cumsum = tf.cond(tf.equal(self.O[Idx], 1), 
                                lambda: _add_to_cumSum(Idx, cumsum),
                                lambda: tf.cast(cumsum, tf.float32))                                    
    
                Idx = tf.cast(tf.add(Idx, 1), tf.int32)
            
            return Idx, cumsum
            
        def _penalty(W):
    
            """
            Elastic net penalty. Inspired by: 
            https://github.com/glm-tools/pyglmnet/blob/master/pyglmnet/pyglmnet.py
            """
            
            with tf.name_scope("Elastic_net"):
                
                # Lasso-like penalty
                L1penalty = tf.reduce_sum(tf.abs(W))
                
                # Compute the L2 penalty (ridge-like)
                L2penalty = tf.reduce_sum(W ** 2)
                    
                # Combine L1 and L2 penalty terms
                P = self.LAMBDA * (self.ALPHA * L1penalty + 0.5 * (1 - self.ALPHA) * L2penalty)
            
            return P
        
        
        with tf.variable_scope("loss"):
    
            cumSum = tf.cast(tf.Variable([0.0]), tf.float32)
            Idx = tf.cast(tf.Variable(0), tf.int32)
            
            # Go through all uncensored cases and add to cumulative sum
            c = lambda Idx, cumSum: tf.less(Idx, tf.cast(tf.size(self.T)-1, tf.int32))
            b = lambda Idx, cumSum: _add_if_observed(Idx, cumSum)
            Idx, cumSum = tf.while_loop(c, b, [Idx, cumSum])
            
            # Add elastic-net penalty
            self.cost = cumSum + _penalty(self.W)

    #%%========================================================================
    #  Optimizer
    #==========================================================================

    def add_optimizer(self):
        
        """
        Adds optimizer to computational graph        
        """
        
        with tf.variable_scope("optimizer"):

            # Define optimizer and minimize loss
            if self.OPTIM == "RMSProp":
                self.optimizer = tf.train.RMSPropOptimizer(self.LEARN_RATE).\
                    minimize(self.cost)
                    
            elif self.OPTIM == "GD":
                self.optimizer = tf.train.GradientDescentOptimizer(self.LEARN_RATE).\
                    minimize(self.cost)
                    
            elif self.OPTIM == "Adam":
                self.optimizer = tf.train.AdamOptimizer(self.LEARN_RATE).\
                    minimize(self.cost)

        # Merge all summaries for tensorboard
        #self.tbsummaries = tf.summary.merge_all()


#%%############################################################################ 
#%%############################################################################ 
#%%############################################################################
#%%############################################################################
