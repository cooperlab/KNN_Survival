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

from scipy.io import loadmat, savemat
import numpy as np
import SurvivalUtils as sUtils
import tensorflow as tf

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
# Build the computational graph
#==============================================================================


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


# Initialize weights and biases
#==============================================================================






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























