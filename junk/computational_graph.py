#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 12:57:55 2017

@author: mohamed
"""

import tensorflow as tf
#import numpy as np

#%%============================================================================
# Computational graph class
#==============================================================================

class comput_graph(object):
    
    """
    Builds the computational graph for Survival NCA.
    """
    
    def __init__(self, dim_input, 
                 transform_type = "linear",
                 nn_params = {'DEPTH': 2}):
        
        """
        Instantiate a computational graph for survival NCA.
        This also adds placeholders.
        
        Args:
        ------
        dim_input - no of features
        
        """
        
        # set up instace attributes
        self.dim_input = dim_input
        
        # clear lurking tensors
        tf.reset_default_graph()
        
        # placeholders
        self._add_placeholders()
        
        # fature space transform
        if transform_type == "linear":
            X_transformed = self.add_linear_transform()
        elif transform_type == "ffNetwork":
            X_transformed = self.add_ffNetwork(**nn_params)
        
        # get Pij
        Pij = self.get_Pij(X_transformed)
            
        


    #%%========================================================================
    # Random useful methods
    #==========================================================================
    
    def _variable_summaries(self, var):
        
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


    #%%========================================================================
    # Add placeholders to graph  
    #==========================================================================
    
    def add_placeholders(self):
    
        """ Adds graph inputs as placeholders in graph """
        
        with tf.variable_scope("Inputs"):
        
            self.X_input = tf.placeholder("float", [None, self.dim_input], name='X_input')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') #for dropout
            
            self.T = tf.placeholder("float", [None], name='T')
            self.O = tf.placeholder("float", [None], name='O')
            self.At_Risk = tf.placeholder("float", [None], name='At_Risk')
            
            
    #%%========================================================================
    # Linear transformation (for interpretability)
    #==========================================================================

    def add_linear_transform(self, Drop = True):
        
        """ 
        Transform features in a linear fashion for better interpretability
        """
        
        with tf.variable_scope("linear_transform"):
            
            # feature scales/weights
            self.w = tf.get_variable("weights", shape=[self.dim_input], 
                            initializer= tf.contrib.layers.xavier_initializer())
            
            self.b = tf.get_variable("biases", shape=[self.dim_input], 
                            initializer= tf.contrib.layers.xavier_initializer())
            
            # diagonalize and matmul
            W = tf.diag(self.w)
            X_transformed = tf.add(tf.matmul(self.X_input, W), self.b) 
            
        return X_transformed
    

    #%%========================================================================
    # Nonlinear feature transformation (feed forward network)
    #==========================================================================

    def add_ffNetwork(self, DEPTH = 2, MAXWIDTH = 200, 
                      NONLIN = "Tanh", LINEAR_READOUT = False,
                      DIM_OUT = None):
        """ 
        Adds a feedforward network to the computational graph,
        which performs a series of non-linear transformations to
        the input matrix and outputs a matrix of the same dimss.
        """
        
        # Define sizes of weights and biases
        #======================================================================
        
        if DIM_OUT is None:
            # do not do any dimensionality reduction
            DIM_OUT = self.dim_input
        
        dim_in = self.dim_input
        
        if DEPTH == 1:
            dim_out = DIM_OUT
        else:
            dim_out = MAXWIDTH
        
        weights_sizes = {'layer_1': [dim_in, dim_out]}
        biases_sizes = {'layer_1': [dim_out]}
        dim_in = dim_out
        
        # intermediate layers
        if DEPTH > 2:
            for i in range(2, DEPTH):                
                dim_out = int(dim_out)
                weights_sizes['layer_{}'.format(i)] = [dim_in, dim_out]
                biases_sizes['layer_{}'.format(i)] = [dim_out]
                dim_in = dim_out
        
        # last layer
        if DEPTH > 1:
            weights_sizes['layer_{}'.format(DEPTH)] = [dim_in, DIM_OUT]
            biases_sizes['layer_{}'.format(DEPTH)] = [DIM_OUT]
        
        # Define a layer
        #======================================================================
        
        def _add_layer(layer_name, Input, APPLY_NONLIN = True,
                       Mode = "Encoder", Drop = True):
            
            """ adds a single fully-connected layer"""
            
            with tf.variable_scope(layer_name):
                #
                # initialize using xavier method
                #
                m_w = weights_sizes[layer_name][0]
                n_w = weights_sizes[layer_name][1]
                m_b = biases_sizes[layer_name][0]
                
                w = tf.get_variable("weights", shape=[m_w, n_w], 
                                    initializer= tf.contrib.layers.xavier_initializer())
                #variable_summaries(w)
             
                b = tf.get_variable("biases", shape=[m_b], 
                                    initializer= tf.contrib.layers.xavier_initializer())
                #variable_summaries(b)
                
                #
                # Do the matmul and apply nonlin
                # 
                if Mode == "Encoder":
                    l = tf.add(tf.matmul(Input, w),b) 
                elif Mode == "Decoder":
                    l = tf.matmul(tf.add(Input,b), w) 
                
                if APPLY_NONLIN:
                    if NONLIN == "Sigmoid":  
                        l = tf.nn.sigmoid(l, name= 'activation')
                    elif NONLIN == "ReLU":  
                        l = tf.nn.relu(l, name= 'activation')
                    elif NONLIN == "Tanh":  
                        l = tf.nn.tanh(l, name= 'activation') 
                    #tf.summary.histogram('activations', l)
                
                # Dropout
                if Drop:
                    l = tf.nn.dropout(l, self.keep_prob)
                    
                return l
        
        # Add the layers
        #======================================================================
            
        with tf.variable_scope("ffNetwork"):
            
            l_in = self.X_input
            
            for i in range(1, DEPTH):
                 l_in = _add_layer("layer_{}".format(i), l_in)
                 
            # outer layer (prediction)
            X_transformed = _add_layer("layer_{}".format(DEPTH), l_in,
                              APPLY_NONLIN = not(LINEAR_READOUT),
                              Drop=False)
            
        return X_transformed


    #%%========================================================================
    # Get Pij 
    #==========================================================================

    def get_Pij(self, X_transformed):
        
        """ 
        Calculate Pij, the probability that j will be chosen 
        as i's neighbor, for all i's
        
        Inspired by: https://github.com/RolT/NCA-python
        """
        
        with tf.name_scope("getting_Pij"):
            # transpose so that feats are in rows
            AX = tf.transpose(X_transformed)
            
            # Expand dims of AX to [n_features, n_samples, n_samples], where
            # each "channel" in the third dimension is the difference between
            # one sample and all other samples
            normAX = AX[:, :, None] - AX[:, None, :]
            
            # Now get the euclidian distance between
            # every patient and all others -> [n_samples, n_samples]
            normAX = tf.norm(normAX, axis=0)
            
            # Calculate Pij, the probability that j will be chosen 
            # as i's neighbor, for all i's
            denomSum = tf.reduce_sum(tf.exp(-normAX), axis=0)
            Pij = tf.exp(-normAX) / denomSum[:, None]
        
        return Pij
    
    
#%%
#%%
#%%

import os
import sys

def conditionalAppend(Dir):
    """ Append dir to sys path"""
    if Dir not in sys.path:
        sys.path.append(Dir)

cwd = os.getcwd()
conditionalAppend(cwd+"/../")
import SurvivalUtils as sUtils

import numpy as np
from scipy.io import loadmat

KEEP_PROB = 0.9

# Load data
dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
#dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Gene.mat"
#dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat"

Data = loadmat(dpath)

Features = np.float32(Data['Integ_X'])
#Features = np.float32(Data['Gene_X'])

N, D = Features.shape

if np.min(Data['Survival']) < 0:
    Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1

Survival = np.int32(Data['Survival']).reshape([N,])
Censored = np.int32(Data['Censored']).reshape([N,])
fnames = Data['Integ_Symbs']
#fnames = Data['Gene_Symbs']

# remove zero-variance features
fvars = np.std(Features, 0)
keep = fvars > 0
Features = Features[:, keep]
fnames = fnames[keep]

# Getting at-risk groups (trainign set)
Features, Survival, Observed, at_risk = \
  sUtils.calc_at_risk(Features, Survival, 1-Censored)

#%%

g = comput_graph(dim_input = D)

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

feed_dict={g.X_input: Features,
           g.T: Survival,
           g.O: Observed,
           g.At_Risk: at_risk,
           g.keep_prob: KEEP_PROB}

#pij = Pij.eval(feed_dict = feed_dict)

