#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 12:57:55 2017

@author: mohamed
"""

import tensorflow as tf


#%%============================================================================
# Computational graph class
#==============================================================================

class computational_graph(object):
    
    """
    Builds the computational graph for Survival NCA.
    """
    
    def __init__(self, dim_input):
        
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
        
        # add placeholders
        self._add_placeholders()


    #%%========================================================================
    # Add placeholders to graph  
    #==========================================================================
    
    def _add_placeholders(self):
    
        """ Adds graph inputs as placeholders in graph """
        
        with tf.variable_scope("Inputs"):
        
            self.X_input = tf.placeholder("float", [None, self.dim_input], name='X_input')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') #for dropout
            
            self.T = tf.placeholder("float", [None], name='T')
            self.O = tf.placeholder("float", [None], name='O')
            self.At_Risk = tf.placeholder("float", [None], name='At_Risk')



    #%%========================================================================
    # Nonlinear feature transformation (feed forward network)
    #==========================================================================

    def add_ffNetwork(self, DEPTH = 2, MAXWIDTH = 200, 
                      NONLIN = "Tanh", LINEAR_READOUT = False):
        """ 
        Adds a feedforward network to the computational graph,
        which performs a series of non-linear transformations to
        the input matrix and outputs a matrix of the same dimss.
        """
        
        # Define sizes of weights and biases
        #======================================================================
        
        dim_in = self.dim_input
        
        if DEPTH == 1:
            dim_out = self.dim_input 
        else:
            dim_out = MAXWIDTH
        
        weights_sizes = {'layer_1': [dim_in, dim_out]}
        biases_sizes = {'layer_1': [dim_out]}
        dim_in = dim_out
        
        if DEPTH > 2:
            for i in range(2, DEPTH):                
                dim_out = int(dim_out)
                weights_sizes['layer_{}'.format(i)] = [dim_in, dim_out]
                biases_sizes['layer_{}'.format(i)] = [dim_out]
                dim_in = dim_out
         
        if DEPTH > 1:
            dim_out = self.dim_input
            weights_sizes['layer_{}'.format(DEPTH)] = [dim_in, dim_out]
            biases_sizes['layer_{}'.format(DEPTH)] = [dim_out]
            dim_in = dim_out
        
        # Define a layer
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
                    if NONLIN == "Sigmoid":  
                        l = tf.nn.sigmoid(l, name= 'activation')
                    elif NONLIN == "ReLU":  
                        l = tf.nn.relu(l, name= 'activation')
                    elif NONLIN == "Tanh":  
                        l = tf.nn.tanh(l, name= 'activation') 
                    #tf.summary.histogram('activations', l)
                
                # Dropout
                
                if Drop:
                    with tf.name_scope('dropout'):
                        l = tf.nn.dropout(l, self.keep_prob)
                    
                return l
        
        # Add the layers
        #======================================================================
            
        with tf.variable_scope("ffNetwork"):
            
            l_in = self.X_input
            
            for i in range(1, DEPTH):
                 l_in = _add_layer("layer_{}".format(i), l_in)
                 
            # outer layer (prediction)
            self.l_out = _add_layer("layer_{}".format(DEPTH), l_in,
                              APPLY_NONLIN = not(LINEAR_READOUT),
                              Drop=False)
            
        return self.l_out


    
    #%%========================================================================
    # Useful tensorflow functions
    #==========================================================================
    
    def _variable_summaries(var):
        
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