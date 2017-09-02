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
MONITOR = True
DISPLAY_STEP = 1
GETLOGS = False
PERC_TRAIN = 0.5
IS_TESTING = False
MODELSAVE_STEP = 10

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








#%%============================================================================
# Build the core network
#==============================================================================





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
    
    cumSum = tf.cond(tf.equal(O[Idx], 1), lambda: Addto_cumSum(Idx, cumSum, t), 
                              lambda: tf.cast(cumSum, tf.float32))
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

c = tf.reduce_sum(cost)    
tf.summary.scalar('cost', c)


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

# Merge all summaries for tensorboard
merged = tf.summary.merge_all()


#%%============================================================================
# Begin session
#==============================================================================

if MONITOR == True and IS_TESTING == False:
    print("Training network...")  
    
# Load weights if existent
LOAD_MODEL = "checkpoint" in os.listdir(WEIGHTPATH)

if not LOAD_MODEL:
    # op to initialise variables
    init = tf.global_variables_initializer()

# op to save/restore all the variables
saver = tf.train.Saver()

# initialize costs
Errors = np.nan * np.ones([EPOCHS, 3])

with tf.Session() as sess:
            
    if LOAD_MODEL:
        print("Restoring saved model ...")
        saver.restore(sess, WEIGHTPATH + "model.ckpt")
        print("Model restored.")
    else:
        # initialize variables
        sess.run(init)
    
    # summary writer for tensorboard
    train_writer = tf.summary.FileWriter(RESULTPATH + '/tensorboard',
                                      sess.graph)
    summary = sess.run(merged, feed_dict={X_input: Features,
                                          T: Survival,
                                          O: Observed,
                                          At_Risk: at_risk,
                                          keep_prob: KEEP_PROB})
    train_writer.add_summary(summary, 0)
    
    
    if IS_TESTING is False:
        try:
            for epoch in range(EPOCHS):
                
                Errors[epoch,0] = epoch 
                
                # train network
                fetches = [optimizer, cost]
                
                _, c = sess.run(fetches, feed_dict={X_input: Features,
                                                    T: Survival,
                                                    O: Observed,
                                                    At_Risk: at_risk,
                                                    keep_prob: KEEP_PROB})
                
                # Add summary for tensorboard
                #train_writer.add_summary(summary, epoch)
    
                # Get training cost
                cost_train = cost.eval(feed_dict={X_input: Features, \
                    T: Survival, O: Observed, At_Risk: at_risk, keep_prob: 1.0})
                Errors[epoch,1] = cost_train
                
                if GETLOGS == True:
                    # Get validation cost
                    cost_valid = cost.eval(feed_dict={X_input: Features_valid, 
                                                      T: Survival_valid, 
                                                      O: Observed_valid, 
                                                      At_Risk: at_risk_valid, 
                                                      keep_prob: 1.0})
                    Errors[epoch,2] = cost_valid
                    
                if MONITOR == True and epoch % DISPLAY_STEP == 0:
                    if GETLOGS == True:
                        print("epoch = {}, cost_train = {},  cost_valid={}"
                              .format(epoch, cost_train, cost_valid))  
                    else:
                        print("epoch = {}, cost_train = {}"
                              .format(epoch, cost_train)) 
                
                
                if ((epoch % MODELSAVE_STEP) == 0):
                    # Save the variables to disk.
                    print("Saving model weights...")
                    save_path = saver.save(sess, WEIGHTPATH + "model.ckpt")
        
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

    
#
# Plot cost
#
if (not IS_TESTING) and MONITOR:
    # plotting errors
    fig, ax = plt.subplots()
    # training error
    ax.plot(Errors[:,0], Errors[:,1], 'b', linewidth=1.5, aa=False)
    
    if GETLOGS:
        # validation error            
        ax.plot(Errors[:,0], Errors[:,2], 'r', linewidth =1.5, aa=False)  
             
    plt.title("Epochs vs -log partial likelihood", fontsize =16, fontweight ='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Cost: Training (blue) / Validation (red)")
    plt.savefig(RESULTPATH + description + "Cost.svg")













