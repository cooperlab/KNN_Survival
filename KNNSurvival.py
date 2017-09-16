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
# KNN model class (trainable model)
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

    def get_neighbor_idxs(self, X_test, X_train, norm = 1):
        
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
        
        # Now get the manhattan or euclidian distance between
        # every patient and all others -> [n_samples, n_samples]
        if norm == 1:
            dist = np.sum(np.abs(dist), axis=2)
        elif norm == 2:
            dist = np.sqrt(np.sum(dist ** 2, axis=2))
        else:
            raise ValueError("Only l1 and l2 norms implemented.")
        
        # Get indices of K nearest neighbors
        neighbor_idxs = np.argsort(dist, axis=1)
        
        return neighbor_idxs
        
        
    #==========================================================================    
        
    def predict(self, neighbor_idxs,
                Survival_train, Censored_train, 
                Survival_test = None, Censored_test = None, 
                K = 15, Method = "cumulative"):
        
        """
        Predict testing set using 'prototype' (i.e. training) set using KNN
        
        neighbor_idxs - indices of nearest neighbors; (N_test, N_train)
        Survival_train - training sample time-to-event; (N,) np array
        Censored_train - training sample censorship status; (N,) np array
        K           - number of nearest-neighbours to use, int
        """
        
        # Keep only desired K
        neighbor_idxs = neighbor_idxs[:, 0:K]

        # Initialize        
        N_test = neighbor_idxs.shape[0]
        T_test = np.zeros([N_test])


        if Method == 'non-cumulative':
            
            # Convert outcomes to "alive status" at each time point 
            alive_train = sUtils.getAliveStatus(Survival_train, Censored_train)
    
            # Get survival prediction for each patient            
            for idx in range(N_test):
                
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
                
        elif Method == 'cumulative':   
            
            for idx in range(N_test):
                
                # Get time and censorship
                T = Survival_train[neighbor_idxs[idx, :]]
                C = Censored_train[neighbor_idxs[idx, :]]
                                
                # find unique times
                t = np.unique(T[C == 0])
                
                # initialize count vectors
                f = np.zeros(t.shape)
                d = np.zeros(t.shape)
                n = np.zeros(t.shape)
                
                # generate counts
                for i in range(len(t)):
                    n[i] = np.sum(T >= t[i])
                    d[i] = np.sum(T[C == 0] == t[i])
                
                # calculate probabilities
                f = (n - d) / n
                f = np.cumprod(f)
                
                # append beginning and end points
                t_start = np.array([0])
                f_start = np.array([1])
                t_end = np.array([T.max()])
                
                t = np.concatenate((t_start, t, t_end), axis=0)
                f = np.concatenate((f_start, f), axis=0)
                
                # now get overall time prediction
                T_test[idx] = np.sum(np.diff(t) * f)
        else:
            raise ValueError("Method is either 'cumulative' or 'non-cumulative'.")
                   
        
        # Get c-index
        CI = 0
        if Survival_test is not None:
            assert (Censored_test is not None)
            CI = sUtils.c_index(T_test, Survival_test, Censored_test, 
                                prediction_type='survival_time')
            
        return T_test, CI



    #%%===========================================================================
    # model tuning and accuracy
    #============================================================================== 

    def cv_tune(self, X, Survival, Censored,
                kcv = 5, shuffles = 1, \
                Ks = list(np.arange(10, 160, 10)),\
                norm = 1, Method = "cumulative"):

        """
        Given an **optimization set**, get optimal K using
        cross-validation with shuffling.
        X - features (n,d)
        Survival - survival (n,)
        Censored - censorship (n,)
        kcv - no of folds for cross validation
        shuffles - no of shuffles for cross validation
        Ks - list of K values to try out
        """

        # Get split indices over optimization set
        splitIdxs = dm.get_balanced_SplitIdxs(Censored, \
                                              K=kcv, SHUFFLES=shuffles,\
                                              USE_OPTIM = False)

        # Initialize
        n_folds = len(splitIdxs['fold_cv_train'][0])
        CIs = np.zeros([n_folds, len(Ks)])
        
        for fold in range(n_folds):
        
            print("\n\tFold {} of {}".format(fold, n_folds-1))
            
            # Isolate patients belonging to fold
        
            idxs_train = splitIdxs['fold_cv_train'][0][fold]
            idxs_test = splitIdxs['fold_cv_test'][0][fold]
            
            X_test = X[idxs_test, :]
            X_train = X[idxs_train, :]
            Survival_train = Survival[idxs_train]
            Censored_train = Censored[idxs_train]
            Survival_test = Survival[idxs_test]
            Censored_test = Censored[idxs_test]
        
            # Get neighbor indices    
            neighbor_idxs = self.get_neighbor_idxs(X_test, X_train, norm = norm)
        
            
            print("\tK \t Ci")
        
            for kidx, K in enumerate(Ks):
            
                # Predict testing set
                _, Ci = self.predict(neighbor_idxs,
                                         Survival_train, Censored_train, 
                                         Survival_test = Survival_test, 
                                         Censored_test = Censored_test, 
                                         K = K, Method = Method)
            
                CIs[fold, kidx] = Ci
            
                print("\t{} \t {}".format(K, round(Ci, 3)))
                             
        
        # Get optimal K
        CIs_median = np.median(CIs, axis=0)
        CI_optim = np.max(CIs_median)
        K_optim = Ks[np.argmax(CIs_median)]
        print("\nOptimal: K = {}, Ci = {}\n".format(K_optim, round(CI_optim, 3)))

        return CIs, K_optim


    #==========================================================================   


    def cv_accuracy(self, X, Survival, Censored, \
                    splitIdxs, outer_fold, \
                    tune_params, \
                    norm = 1, Method = 'cumulative'):

        """
        Find model accuracy using KCV (after ptimizing K)
        
        X - features (n,d)
        Survival - survival (n,)
        Censored - censorship (n,)
        splitIdxs - dict; indices of patients belonging to each fold
        outer_fold - fold index for optimization and non-optim. sets
        tune_params - dict; parameters to pass to cv_tune method
        """

        # Initialize
        n_folds = len(splitIdxs['fold_cv_train'][0])
        CIs = np.zeros([n_folds])

        print("\nOptimizing K for this outer fold.")

        # find optimal K on validation set
        optimIdxs = splitIdxs['idx_optim'][outer_fold]

        _, K_optim = self.cv_tune(X[optimIdxs, :], \
                                  Survival[optimIdxs], \
                                  Censored[optimIdxs], \
                                  **tune_params)
        
        print("outer_fold \t fold \t Ci")

        for fold in range(n_folds):
        
            # Isolate patients belonging to fold
        
            idxs_train = splitIdxs['fold_cv_train'][outer_fold][fold]
            idxs_test = splitIdxs['fold_cv_test'][outer_fold][fold]
            
            X_test = X[idxs_test, :]
            X_train = X[idxs_train, :]
            Survival_train = Survival[idxs_train]
            Censored_train = Censored[idxs_train]
            Survival_test = Survival[idxs_test]
            Censored_test = Censored[idxs_test]
        
            # Get neighbor indices    
            neighbor_idxs = self.get_neighbor_idxs(X_test, X_train, norm = norm)
        
            # Predict testing set
            _, Ci = self.predict(neighbor_idxs,
                                 Survival_train, Censored_train, 
                                 Survival_test = Survival_test, 
                                 Censored_test = Censored_test, 
                                 K = K_optim, Method = Method)
            
            CIs[fold] = Ci
               
            print("{} \t {} \t {}".format(outer_fold, fold, round(Ci, 3)))


        return CIs, K_optim
