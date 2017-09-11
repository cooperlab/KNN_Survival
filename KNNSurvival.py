#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 00:16:36 2017

@author: mohamed

Predict survival using KNN
"""

# Append relevant paths
import os
import sys

def conditionalAppend(Dir):
    """ Append dir to sys path"""
    if Dir not in sys.path:
        sys.path.append(Dir)

cwd = os.getcwd()
conditionalAppend(cwd)

#import _pickle
import numpy as np
#from matplotlib import cm
#import matplotlib.pylab as plt

import logging
import datetime

import ProjectUtils as pUtils
import SurvivalUtils as sUtils
#import DataManagement as dm

#raise(Exception)

#%%============================================================================
# NCAmodel class (trainable model)
#==============================================================================

class SurvivalKNN(object):
    
    """
    KNN Survival.
    
    This predicts survival based on the labels of K-nearest neighbours 
    using weighted euclidian distance and non-cumulative survival probability.
        
    """
    
    # Init
    ###########################################################################
    
    def __init__(self, 
                 RESULTPATH, description=""):
        
        """Instantiate a survival NCA object"""
        
        # Set instance attributes
        #==================================================================
        
        self.RESULTPATH = RESULTPATH
        self.LOGPATH = self.RESULTPATH + "logs/"
        
        # prefix to all saved results
        self.description = description
        
        
        # Create output dirs
        #==================================================================
        
        self._makeSubdirs()
        
        # Configure logger - will not work with iPython
        #==================================================================
        
        timestamp = str(datetime.datetime.today()).replace(' ','_')
        logging.basicConfig(filename = self.LOGPATH + timestamp + "_RunLogs.log", 
                            level = logging.INFO,
                            format = '%(levelname)s:%(message)s')
                            
    #%%===========================================================================
    # Miscellaneous methods
    #==============================================================================
    
    def getModelInfo(self):
        
        """ Returns relevant model attributes"""
        
        attribs = {
            'RESULTPATH' : self.RESULTPATH,
            'description' : self.description,
            'LOGPATH': self.LOGPATH,
            }
        
        return attribs

    #==========================================================================    
    
    def _makeSubdirs(self):
        
        """ Create output directories"""
        
        # Create relevant result subdirectories
        pUtils.makeSubdir(self.RESULTPATH, 'plots')
        pUtils.makeSubdir(self.RESULTPATH, 'preds')
        pUtils.makeSubdir(self.RESULTPATH, 'logs')
        

    #%%===========================================================================
    # Actual prediction model
    #==============================================================================        
        
    def predict(self, X_test, X_train, 
                Survival_train, Censored_train, 
                Survival_test = None, Censored_test = None, 
                K = 15):
        
        """
        Predict testing set using 'prototype' (i.e. training) set using KNN
        
        Args: 
        --------
        WARNING:   Unless preceeded by NCA, input data has to be normalized
        to have similar scales
         
        X_test      - testing sample features; (N, D) np array
        X_train     - training sample features; (N, D) np array
        Survival_train - training sample time-to-event; (N,) np array
        Censored_train - training sample censorship status; (N,) np array
        K           - number of nearest-neighbours to use, int
        """

        # Convert outcomes to "alive status" at each time point
        #======================================================================
        
        pUtils.Log_and_print("Getting survival status.")
        
        alive_train = sUtils.getAliveStatus(Survival_train, Censored_train)
        Survival_train = None
        Censored_train = None

        # Find nearest neighbors
        #======================================================================

        pUtils.Log_and_print("Finding the {} nearest neighbors.".format(K))
        
        # Expand dims of AX to [n_samples_test, n_samples_train, n_features], 
        # where each "channel" in the third dimension is the difference between
        # one testing sample and all training samples along one feature
        dist = X_train[None, :, :] - X_test[:, None, :]
        
        # Now get the euclidian distance between
        # every patient and all others -> [n_samples, n_samples]
        #normAX = tf.norm(normAX, axis=0)
        dist = np.sqrt(np.sum(dist ** 2, axis=2))
        
        # Get indices of K nearest neighbors
        neighbor_idxs = np.argsort(dist, axis=1)[:, 0:K]
        
        # Get survival prediction for each patient
        #======================================================================
        
        pUtils.Log_and_print("Making predictions.")
        
        T_test = np.zeros([X_test.shape[0]])
        
        for idx in range(T_test.shape[0]):
            
            status = alive_train[neighbor_idxs[idx, :], :]
            totalKnown = np.sum(status >= 0, axis = 0)
            status[status < 0] = 0
            
            # remove timepoints where there are no known statuses
            status = status[:, totalKnown != 0]
            totalKnown = totalKnown[totalKnown != 0]
            
            # get "average" predicted survival time
            status = np.sum(status, axis = 0) / totalKnown
            
            # now get overall time prediction            
            T_test[idx] = np.sum(status)
            #T_test[idx] = np.mean(status)
            #T_test[idx] = 1 - np.mean(status)
        
        # Get c-index
        #======================================================================
        CI = 0
        if Survival_test is not None:
            assert (Censored_test is not None)
            CI = sUtils.c_index(T_test, Survival_test, Censored_test, 
                                prediction_type='survival_time')
            
        return T_test, CI