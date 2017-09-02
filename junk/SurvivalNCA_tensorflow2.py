#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 18:42:02 2017

@author: mohamed
"""
# Append relevant paths
import os
import sys

def conditionalAppend(Dir):
    """ Append dir to sys path"""
    if Dir not in sys.path:
        sys.path.append(Dir)

cwd = os.getcwd()
conditionalAppend(cwd+"/../")

from scipy.io import loadmat #, savemat
import numpy as np
import SurvivalUtils as sUtils
import tensorflow as tf
import matplotlib.pylab as plt

#%%============================================================================
# Load and preprocess data
#==============================================================================

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

#%%============================================================================
# Function params
#==============================================================================

EPOCHS = 3000
NONLIN = "Tanh"
OPTIM = "RMSProp"
LEARN_RATE = 0.01
KEEP_PROB = 0.9
DEPTH = 2
MAXWIDTH = 200
MONITOR = True
DISPLAY_STEP = 1
GETLOGS = False
PERC_TRAIN = 0.5
LINEAR_READOUT = False
IS_TESTING = False

RESULTPATH = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/tmp/"
description = "GBMLGG_Integ_"

#%%============================================================================
# Basic data management
#==============================================================================

if GETLOGS == True:
    # Separate out validation set
    N_tot = np.size(Features, 0)
    Features_valid = Features[int(PERC_TRAIN * N_tot):N_tot,:]
    Survival_valid = Survival[int(PERC_TRAIN * N_tot):N_tot]
    Censored_valid = Censored[int(PERC_TRAIN * N_tot):N_tot]
    
    Features = Features[0:int(PERC_TRAIN * N_tot),:]
    Survival = Survival[0:int(PERC_TRAIN * N_tot)]
    Censored = Censored[0:int(PERC_TRAIN * N_tot)]
    
    # Getting at-risk groups (validation set)
    Features_valid, Survival_valid, Observed_valid, at_risk_valid = \
      sUtils.calc_at_risk(Features_valid, Survival_valid, 1-Censored_valid)

# Getting at-risk groups (trainign set)
Features, Survival, Observed, at_risk = \
  sUtils.calc_at_risk(Features, Survival, 1-Censored)
        

#%%============================================================================
# Setting params and other stuff
#==============================================================================
    
# Convert to integer/bool (important for BayesOpt to work properly since it 
# tries float values)    
EPOCHS = int(EPOCHS)       
DEPTH = int(DEPTH)
MAXWIDTH = int(MAXWIDTH)
    
# Get size of features
dim_input = Features.shape[1]

# Create dir to save weigths
WEIGHTPATH = RESULTPATH + description + "weights/"
os.system("mkdir " + WEIGHTPATH)


# Define sizes of weights and biases
#==============================================================================

dim_in = dim_input

if DEPTH == 1:
    dim_out = 1 
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
    dim_out = 1
    weights_sizes['layer_{}'.format(DEPTH)] = [dim_in, dim_out]
    biases_sizes['layer_{}'.format(DEPTH)] = [dim_out]
    dim_in = dim_out


# Useful tensorflow functions
#==============================================================================

def variable_summaries(var):
    
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


#%%============================================================================
# Build the core network
#==============================================================================

# clear lurking tensors
tf.reset_default_graph()


# Placeholders
#==============================================================================

with tf.variable_scope("Inputs"):
    
    X_input = tf.placeholder("float", [None, dim_input])
    keep_prob = tf.placeholder(tf.float32) #for dropout
    
    T = tf.placeholder("float", [None])
    O = tf.placeholder("float", [None])
    At_Risk = tf.placeholder("float", [None])


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
        
        with tf.name_scope("weights"):
            w = tf.get_variable("weights", shape=[m_w, n_w], initializer= xavier)
            variable_summaries(w)
     
        with tf.name_scope("biases"):
            b = tf.get_variable("biases", shape=[m_b], initializer= xavier)
            variable_summaries(b)
            
        # Do the matmul and apply nonlin
        
        with tf.name_scope("pre_activations"):   
            if Mode == "Encoder":
                l = tf.add(tf.matmul(Input, w),b) 
            elif Mode == "Decoder":
                l = tf.matmul(tf.add(Input,b), w) 
            tf.summary.histogram('pre_activations', l)
        
        if APPLY_NONLIN:
            if NONLIN == "Sigmoid":  
                l = tf.nn.sigmoid(l, name= 'activation')
            elif NONLIN == "ReLU":  
                l = tf.nn.relu(l, name= 'activation')
            elif NONLIN == "Tanh":  
                l = tf.nn.tanh(l, name= 'activation') 
            tf.summary.histogram('activations', l)
        
        # Dropout
        
        if Drop:
            with tf.name_scope('dropout'):
                l = tf.nn.dropout(l, keep_prob)
            
        return l

#
# Now add the layers
#
    
with tf.variable_scope("FFNetwork"):
    
    l_in = X_input
    
    for i in range(1, DEPTH):
         l_in = _add_layer("layer_{}".format(i), l_in)
         
    # outer layer (prediction)
    l_out = _add_layer("layer_{}".format(DEPTH), l_in,
                      APPLY_NONLIN = not(LINEAR_READOUT),
                      Drop=False)


## Merge all the summaries and write them out to /tmp/mnist_logs (by default)
#merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
#                                      sess.graph)

#%%============================================================================
# Define loss and optimizer      
#==============================================================================

def Addto_cumSum(Idx, cumSum, t):
    
    """Add patient to log partial likelihood sum """
    
    Idx = tf.cast(Idx, tf.int32)
    cumSum = tf.cast(cumSum, tf.float32)
    t = tf.cast(t, tf.float32)
    
    # Get X*B of current patient and corresponding at-risk cases
    # i.e. those with higher survival or last follow-up time
    Pred_ThisPatient = t[Idx]
    Pred_AtRisk = t[tf.cast(At_Risk[Idx], tf.int32):tf.size(t)-1]
    
    # Get log partial sum of prediction for those at risk
    LogPartialSum = tf.log(tf.reduce_sum(tf.exp(Pred_AtRisk)))
    
    # Get difference
    Diff_ThisPatient = tf.subtract(Pred_ThisPatient, LogPartialSum)
    
    # Add to cumulative log partial likeliood sum
    cumSum = tf.add(cumSum, Diff_ThisPatient)
    cumSum = tf.cast(cumSum, tf.float32)
    
    return cumSum

def Conditional_Addto_cumSum(Idx, cumSum, t):
    
    """ Add to cumsum if current patient'd death time is observed """
    
    cumSum = tf.cond(tf.equal(O[Idx], 1), lambda: Addto_cumSum(Idx, cumSum, t), lambda: tf.cast(cumSum, tf.float32))
    Idx = tf.add(Idx, 1)
    return Idx, cumSum


# Calculate log partial likelihood
#==============================================================================

with tf.variable_scope("loss"):

    cumSum_Pred = tf.Variable([0.0])
    Idx = tf.constant(0)
    
    # Doing the following admittedly odd step because tensorflow's loop
    # requires both the condition and body to have same number of inputs
    def cmp_pred(Idx, cumSum_Pred):
        return tf.less(tf.cast(Idx, tf.int32), tf.cast(tf.size(l_out)-1, tf.int32))
    
    # Go through all uncensored cases and add to cumulative sum
    c = lambda Idx, cumSum: cmp_pred(Idx, cumSum)
    b = lambda Idx, cumSum: Conditional_Addto_cumSum(Idx, cumSum, l_out)
    Idx, cumSum_Pred = tf.while_loop(c, b, [Idx, cumSum_Pred])
    
    
    # cost is negative log patial likelihood
    cost = -cumSum_Pred


# Optimizer
#==============================================================================

with tf.variable_scope("optimizer"):
    
    # Define optimizer and minimize loss
    if OPTIM == "RMSProp":
        optimizer = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(cost)
    elif OPTIM == "GD":
        optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(cost)
    elif OPTIM == "Adam":
        optimizer = tf.train.AdamOptimizer(LEARN_RATE).minimize(cost)


#%%============================================================================
# Begin session
#==============================================================================


## Load weights if existent
#LOAD_MODEL = "checkpoint" in os.listdir(WEIGHTPATH)
#
#if not LOAD_MODEL:
#    # op to initialise variables
#    init = tf.global_variables_initializer()
#
## op to save/restore all the variables
#saver = tf.train.Saver()
#
#
#with tf.Session() as sess:
#            
#    if LOAD_MODEL:
#        print("Restoring saved model ...")
#        saver.restore(sess, WEIGHTPATH + "model.ckpt")
#        print("Model restored.")
#    else:
#        # initialize variables
#        sess.run(init)


#%%============================================================================
# 
#==============================================================================

if MONITOR == True and IS_TESTING == False:
    print("Training network...")        

Errors = np.nan * np.ones([EPOCHS, 3])

# Launch the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    if IS_TESTING is False:
        try:
            for epoch in range(EPOCHS):
                
                Errors[epoch,0] = epoch 
                
                # train network
                _, c = sess.run([optimizer, cost], feed_dict={X_input: Features, \
                    T: Survival, O: Observed, At_Risk: at_risk, keep_prob: KEEP_PROB})
                    
                # Get training cost
                cost_train = cost.eval(feed_dict={X_input: Features, \
                    T: Survival, O: Observed, At_Risk: at_risk, keep_prob: 1.0})
                Errors[epoch,1] = cost_train
                
                if GETLOGS == True:
                    # Get validation cost
                    cost_valid = cost.eval(feed_dict={X_input: Features_valid, \
                        T: Survival_valid, O: Observed_valid, At_Risk: at_risk_valid, keep_prob: 1.0})
                    Errors[epoch,2] = cost_valid
                    
                if MONITOR == True and epoch % DISPLAY_STEP == 0:
                    if GETLOGS == True:
                        print("epoch = {}, cost_train = {},  cost_valid={}".format(epoch, cost_train, cost_valid))  
                    else:
                        print("epoch = {}, cost_train = {}".format(epoch, cost_train))  
        
        except KeyboardInterrupt:
            # Get rid of unused initialized errors if KeyBoardInterrupt
            Errors = Errors[0:epoch,:]
    
        if MONITOR == True:    
            print("Optimization Finished!")
            print("\nFetching final results...")
            
        if GETLOGS == True:
            # merging training and validation sets back
            Features = np.concatenate((Features, Features_valid), axis=0)
            Survival = np.concatenate((Survival, Survival_valid), axis=0)
            Observed = np.concatenate((Observed, Observed_valid), axis=0)
            at_risk = np.concatenate((at_risk, at_risk_valid), axis=0)
        
    # Fetching final results
    risk = l_out.eval(feed_dict={X_input: Features, \
                    T: Survival, O: Observed, At_Risk: at_risk, keep_prob: 1.0})

if IS_TESTING == False and GETLOGS == True and MONITOR == True:
    # plotting errors
    fig, ax = plt.subplots()
    # training error
    ax.plot(Errors[:,0], Errors[:,1], 'b', linewidth=1.5, aa=False)
    # validation error            
    ax.plot(Errors[:,0], Errors[:,2], 'r', linewidth =1.5, aa=False)  
             
    plt.title("Epochs vs -log partial likelihood", fontsize =16, fontweight ='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Cost: Training (blue) / Validation (red)")        

# Getting concordance index
c = sUtils.c_index(risk, Survival, 1-Observed)
if MONITOR == True:        
    print("c_index = {}".format(c))

Outputs = {'c_index': c,
          'risk': risk,
          'Errors': Errors,}














