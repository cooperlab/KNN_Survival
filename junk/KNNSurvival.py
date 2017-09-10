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
        
    def predict(self, X_test, 
                X_train, 
                Survival_train, Censored_train, K):
        
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
        
        