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

#import logging
#import datetime

import ProjectUtils as pUtils
import SurvivalUtils as sUtils
import DataManagement as dm

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
        
                            
    #%%===========================================================================
    # Miscellaneous methods
    #==============================================================================
    
    def getModelInfo(self):
        
        """ Returns relevant model attributes"""
        
        attribs = {
            'RESULTPATH' : self.RESULTPATH,
            'description' : self.description,
            }
        
        return attribs

    #==========================================================================    
    
    def _makeSubdirs(self):
        
        """ Create output directories"""
        
        # Create relevant result subdirectories
        pUtils.makeSubdir(self.RESULTPATH, 'plots')
        pUtils.makeSubdir(self.RESULTPATH, 'preds')
        

    #%%===========================================================================
    # Actual prediction model
    #==============================================================================

    def get_neighbor_idxs(self, X_test, X_train):
        
        """ 
        Get indices of nearest neighbors.
        
        Args: 
        --------
        WARNING:   Unless preceeded by NCA, input data has to be normalized
        to have similar scales
         
        X_test      - testing sample features; (N, D) np array
        X_train     - training sample features; (N, D) np array
        
        """
        
        # Expand dims of AX to [n_samples_test, n_samples_train, n_features], 
        # where each "channel" in the third dimension is the difference between
        # one testing sample and all training samples along one feature
        dist = X_train[None, :, :] - X_test[:, None, :]
        
        # Now get the euclidian distance between
        # every patient and all others -> [n_samples, n_samples]
        #normAX = tf.norm(normAX, axis=0)
        dist = np.sqrt(np.sum(dist ** 2, axis=2))
        
        # Get indices of K nearest neighbors
        neighbor_idxs = np.argsort(dist, axis=1)
        
        return neighbor_idxs
        
        
    #==========================================================================    
        
    def predict(self, neighbor_idxs,
                Survival_train, Censored_train, 
                Survival_test = None, Censored_test = None, 
                K = 15):
        
        """
        Predict testing set using 'prototype' (i.e. training) set using KNN
        
        neighbor_idxs - indices of nearest neighbors; (N_test, N_train)
        Survival_train - training sample time-to-event; (N,) np array
        Censored_train - training sample censorship status; (N,) np array
        K           - number of nearest-neighbours to use, int
        Method      - cumulative vs non-cumulative probability
        """
        
        # Keep only desired K
        neighbor_idxs = neighbor_idxs[:, 0:K]

        # Initialize        
        N_test = neighbor_idxs.shape[0]
        T_test = np.zeros([N_test])

        for idx in range(N_test):
            
            # Get at-risk groups for each time point for nearest neighbors
            T = Survival_train[neighbor_idxs[idx, :]]
            O = 1 - Censored_train[neighbor_idxs[idx, :]]
            T, O, at_risk, _ = sUtils.calc_at_risk(T, O)
            
            N_at_risk = K - at_risk
            
            # Calcuate cumulative probability of survival
            P = np.cumprod((N_at_risk - O) / N_at_risk)
            
            # now get overall time prediction
            T_test[idx] = np.sum(P)
                   
        
        # Get c-index
        CI = 0
        if Survival_test is not None:
            assert (Censored_test is not None)
            CI = sUtils.c_index(T_test, Survival_test, Censored_test, 
                                prediction_type='survival_time')
            
        return T_test, CI



    #%%===========================================================================
    # model tuning
    #============================================================================== 

    def cv_tune(self, X, Survival, Censored,
                kcv = 5, shuffles = 5, \
                Ks = list(np.arange(10, 160, 10))):

        """
        Given an optimization set, get optimal K using
        cross-validation with shuffling.
        X - features (n,d)
        Survival - survival (n,)
        Censored - censorship (n,)
        kcv - no of folds for cross validation
        shuffles - no of shuffles for cross validation
        Ks - list of K values to try out
        """

        # Get split indices over optimization set
        splitIdxs = dm.get_balanced_SplitIdxs(Censored, OPTIM_RATIO=0, \
                                              K=kcv, SHUFFLES=shuffles)

        # Initialize
        n_folds = len(splitIdxs['fold_cv_train'])
        CIs = np.zeros([n_folds, len(Ks)])
        
        for fold in range(n_folds):
        
            print("\nFold {} of {}".format(fold, n_folds-1))
            print("----------------------------------------\n")
            
            # Isolate patients belonging to fold
        
            idxs_train = splitIdxs['fold_cv_train'][fold]
            idxs_test = splitIdxs['fold_cv_test'][fold]
            
            X_test = X[idxs_test, :]
            X_train = X[idxs_train, :]
            Survival_train = Survival[idxs_train]
            Censored_train = Censored[idxs_train]
            Survival_test = Survival[idxs_test]
            Censored_test = Censored[idxs_test]
        
            # Get neighbor indices    
            neighbor_idxs = self.get_neighbor_idxs(X_test, X_train)
        
            
            print("K \t Ci")
        
            for kidx, K in enumerate(Ks):
            
                # Predict testing set
                _, Ci = self.predict(neighbor_idxs,
                                         Survival_train, Censored_train, 
                                         Survival_test = Survival_test, 
                                         Censored_test = Censored_test, 
                                         K = K)
            
                CIs[fold, kidx] = Ci
            
                print("{} \t {}".format(K, round(Ci, 3)))
                             
        
        # Get optimal K
        CIs_mean = np.mean(CIs, axis=0)
        CI_optim = np.max(CIs_mean)
        K_optim = Ks[np.argmax(CIs_mean)]
        print("Optimal: K = {}, Ci = {}".format(K_optim, round(CI_optim, 3)))

        return CIs, K_optim

#
