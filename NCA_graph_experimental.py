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
                 OPTIM = 'GD',
                 LEARN_RATE = 0.01,
                 per_split_feats = 500,
                 transform = 'linear',
                 regularization = 'L2',
                 dim_output = 100,
                 ROTATE = False,
                 DEPTH = 2,
                 MAXWIDTH = 200,
                 NONLIN = 'Tanh'):
        
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
        
        # set up instace attributes
        # =====================================================================
        
        self.dim_input = dim_input
        
        if dim_output > dim_input:
            dim_output = dim_input
        self.dim_output = int(dim_output)
        
        self.OPTIM = OPTIM
        self.LEARN_RATE = LEARN_RATE
        self.per_split_feats = per_split_feats

        self.transform = transform
        self.regularization = regularization
        
        if self.transform == 'linear':
            
            if not ROTATE:        
                assert dim_output == dim_input

            # linear transform params
            self.ROTATE = ROTATE
            
        elif self.transform == 'ffnn':

            # feed forward network architecture
            self.DEPTH = int(DEPTH)
            self.MAXWIDTH = int(MAXWIDTH)
            self.NONLIN = NONLIN
                   
        else:
            raise ValueError("Unknown transform type.")
           
        # Add computational graph
        # =====================================================================
        
        tf.reset_default_graph()
        self.add_placeholders()
        
        if self.transform == 'linear':
            self.add_linear_transform()
        elif self.transform == 'ffnn':
            self.add_ffnn()
            
        self.add_cost()
        self.add_optimizer()
        

    #%%========================================================================
    # Add placeholders to graph  
    #==========================================================================
    
    def add_placeholders(self):
    
        """ Adds graph inputs as placeholders in graph """
        
        with tf.variable_scope("Inputs"):
            
            # Inputs
            self.X_input = tf.placeholder("float", [None, self.dim_input], name='X_input')
            self.Pij_mask = tf.placeholder("float", [None, None], name='Pij_mask')
            
            # Hyperparams
            self.ALPHA = tf.placeholder(tf.float32, name='ALPHA')
            self.LAMBDA = tf.placeholder(tf.float32, name='LAMDBDA')
            self.SIGMA = tf.placeholder(tf.float32, name='SIGMA')
            self.DROPOUT_FRACTION = tf.placeholder(tf.float32, name='DROPOUT_FRACTION')
            
                 
    #%%========================================================================
    # Linear transformation
    #==========================================================================

    def add_linear_transform(self):
        
        """ 
        Transform features in a linear fashion for better interpretability
        """
        
        with tf.variable_scope("linear_transform"):
            
            
            if self.ROTATE:
                self.W = tf.get_variable("weights", shape=[self.dim_input, self.dim_output], 
                                initializer= tf.contrib.layers.xavier_initializer())

            else:
                
                # Initialize weights to a slightly noisy identity matrix
                epsilon = 0.5
                weights_init = 1 + tf.random_uniform(shape=(self.dim_input, ), 
                                                     minval= -epsilon, maxval= epsilon, 
                                                     dtype= tf.float32)
                
                # feature scales/weights
                #self.w = tf.get_variable("weights", shape=[self.dim_input], 
                #                initializer= tf.contrib.layers.xavier_initializer())
                self.w = tf.get_variable("weights", initializer= weights_init)
                
                # diagonalize and matmul
                self.W = tf.diag(self.w)
            
            self.X_transformed = tf.matmul(self.X_input, self.W)

            
    #%%========================================================================
    # Feed-forward neural network transform
    #==========================================================================

    def add_ffnn(self):
        
        """
        Transform features non-linearly using a feed-forward neural network
        with potential dimensionality reduction
        """

        # Define sizes of weights and biases
        #======================================================================
        
        dim_in = self.dim_input
        
        if self.DEPTH == 1:
            dim_out = self.dim_output
        else:
            dim_out = self.MAXWIDTH
        
        weights_sizes = {'layer_1': [dim_in, dim_out]}
        biases_sizes = {'layer_1': [dim_out]}
        dim_in = dim_out
        
        if self.DEPTH > 2:
            for i in range(2, self.DEPTH):                
                dim_out = int(dim_out)
                weights_sizes['layer_{}'.format(i)] = [dim_in, dim_out]
                biases_sizes['layer_{}'.format(i)] = [dim_out]
                dim_in = dim_out
         
        if self.DEPTH > 1:
            dim_out = self.dim_output
            weights_sizes['layer_{}'.format(self.DEPTH)] = [dim_in, dim_out]
            biases_sizes['layer_{}'.format(self.DEPTH)] = [dim_out]
            dim_in = dim_out
            
            
        # Define layers
        #======================================================================
        
        def _add_layer(layer_name, Input, APPLY_NONLIN = True,
                       Mode = "Encoder", Drop = True):
            
            """ adds a single fully-connected layer"""
            
            with tf.variable_scope(layer_name):
                
                # initialize using xavier method
                
                m_w = weights_sizes[layer_name][0]
                n_w = weights_sizes[layer_name][1]
                m_b = biases_sizes[layer_name][0]
                
                xavier = tf.contrib.layers.xavier_initializer()
                
                w = tf.get_variable("weights", shape=[m_w, n_w], initializer= xavier)
                #variable_summaries(w)
             
                b = tf.get_variable("biases", shape=[m_b], initializer= xavier)
                #variable_summaries(b)
                    
                # Do the matmul and apply nonlin
                
                with tf.name_scope("pre_activations"):   
                    if Mode == "Encoder":
                        l = tf.add(tf.matmul(Input, w),b) 
                    elif Mode == "Decoder":
                        l = tf.matmul(tf.add(Input,b), w) 
                    #tf.summary.histogram('pre_activations', l)
                
                if APPLY_NONLIN:
                    if self.NONLIN == "Sigmoid":  
                        l = tf.nn.sigmoid(l, name= 'activation')
                    elif self.NONLIN == "ReLU":  
                        l = tf.nn.relu(l, name= 'activation')
                    elif self.NONLIN == "Tanh":  
                        l = tf.nn.tanh(l, name= 'activation') 
                    #tf.summary.histogram('activations', l)
                
                # Dropout
                
                if Drop:
                    with tf.name_scope('dropout'):
                        l = tf.nn.dropout(l, keep_prob= 1-self.DROPOUT_FRACTION)
                    
                return l
        
        # Now add the layers
        #======================================================================
            
        with tf.variable_scope("FFNetwork"):
            
            l_in = self.X_input
         
            layer_params = {'APPLY_NONLIN' : True,
                            'Mode' : "Encoder",
                            'Drop' : True,
                            }
                       
            for i in range(1, self.DEPTH):
                 l_in = _add_layer("layer_{}".format(i), l_in, **layer_params)
                 
            # outer layer (final, transformed datset)
            layer_params['Drop'] = False
            self.X_transformed = _add_layer("layer_{}".format(self.DEPTH), l_in, **layer_params)            
    
    
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
                return tf.exp(tf.multiply(-z, self.SIGMA))

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
        
            
        def _penalty(W):
    
            """
            Elastic net penalty. Inspired by: 
            https://github.com/glm-tools/pyglmnet/blob/master/pyglmnet/pyglmnet.py
            """
            
            with tf.name_scope("Regularization"):
                
                if self.regularization in ['L1', 'elasticnet']:
                    # Lasso-like penalty
                    L1penalty = tf.reduce_sum(tf.abs(W))
                
                if self.regularization in ['L2', 'elasticnet']:
                    # Compute the L2 penalty (ridge-like)
                    L2penalty = tf.reduce_sum(W ** 2)
                
                if self.regularization == 'L1':
                    P = L1penalty
                elif self.regularization == 'L2':
                    P = L2penalty
                else:
                    # Combine L1 and L2 penalty terms (elastic net)
                    P = self.LAMBDA * (self.ALPHA * L1penalty + (1 - self.ALPHA) * L2penalty)
            
            return P
        
        
        with tf.variable_scope("loss"):
            
            # Multiply Pij by weighted/unweighted mask
            self.Pij = tf.multiply(self.Pij, self.Pij_mask)
            
            # cost the sum of Pij over all observed cases
            self.cost = tf.reduce_sum(self.Pij)
            
            if self.transform == 'linear':            
                self.cost = self.cost + _penalty(self.W)


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
                    
            elif self.OPTIM == "FTRL":
                # Coordinate descent
                self.optimizer = tf.train.FtrlOptimizer(self.LEARN_RATE).\
                    minimize(self.cost)

        # Merge all summaries for tensorboard
        #self.tbsummaries = tf.summary.merge_all()


#%%############################################################################ 
#%%############################################################################ 
#%%############################################################################
#%%############################################################################
