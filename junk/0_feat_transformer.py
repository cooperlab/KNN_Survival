#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 12:40:23 2017

@author: mohamed
"""

import tensorflow as tf





#%%

DEPTH = 2
MAXWIDTH = 200
NONLIN = "Tanh"
LINEAR_READOUT = False




#%%
# Define sizes of weights and biases
#==============================================================================

dim_in = dim_input

if DEPTH == 1:
    dim_out = dim_input 
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
    dim_out = dim_input
    weights_sizes['layer_{}'.format(DEPTH)] = [dim_in, dim_out]
    biases_sizes['layer_{}'.format(DEPTH)] = [dim_out]
    dim_in = dim_out


# Actual layers
#==============================================================================

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
                l = tf.nn.dropout(l, keep_prob)
            
        return l


# Now add the layers
    
with tf.variable_scope("FFNetwork"):
    
    l_in = X_input
    
    for i in range(1, DEPTH):
         l_in = _add_layer("layer_{}".format(i), l_in)
         
    # outer layer (prediction)
    l_out = _add_layer("layer_{}".format(DEPTH), l_in,
                      APPLY_NONLIN = not(LINEAR_READOUT),
                      Drop=False)