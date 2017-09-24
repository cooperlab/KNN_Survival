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
        
        """Instantiate a survival KNN object"""
        
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
    # Supporting methods
    #==============================================================================

    def _get_neighbor_idxs(self, X_test, X_train, norm = 2):
        
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

    def _get_events(self, T, C):

        """
        Get unique times, num-at-risk ad num-of-events.

        Args:
        -----
        T - time-to-event
        C - Censorship indicator

        Returns:
        --------
        t - unique times
        n - no. at risk
        d - no of observed events
        """

        # find unique times
        t = np.unique(T[C == 0])
        
        # initialize count vectors
        d = np.zeros(t.shape)
        n = np.zeros(t.shape)
        
        # generate counts
        for i in range(len(t)):
            n[i] = np.sum(T >= t[i]) # <- no at risk
            d[i] = np.sum(T[C == 0] == t[i]) # <- no of events

        return t, n, d

    #==========================================================================    

    def _km_estimator(self, T, C):
        
        """
        Get kaplan-meier survivor function. 
        
        Args:
        -----
        T - time-to-event (or last-follow-up)
        C - censoring indicator
        
        Returns:
        --------
        t, f - time and survival probability
        """
        
        # get event counts for unique times
        t, n, d = self._get_events(T, C)
       
        # calculate probabilities
        f = (n - d) / n
        f = np.cumprod(f)
        
        # append beginning and end points
        t_start = np.array([0])
        f_start = np.array([1])
        t_end = np.array([T.max()])

        f_end = np.array([f[-1]])
        
        # Get estimate of K-M survivor function
        t = np.concatenate((t_start, t, t_end), axis=0)
        f = np.concatenate((f_start, f, f_end), axis=0)

        return t, f
        
    #==========================================================================    

    def _na_estimator(self, T, C):
        
        """
        Get Nelson-Aalen estimator of risk
        See: www.med.mcgill.ca/epidemiology/hanley/material/ ...
             NelsonAalenEstimator.pdf
        Args:
        -----
        T - time-to-event (or last-follow-up)
        C - censoring indicator
        
        Returns:
        --------
        t, f - time and cumulative risk
        """
        
        # get event counts for unique times
        t, n, d = self._get_events(T, C)

        # calculate hazard rates
        f = d / n
        f = np.cumsum(f)
        
        # append beginning and end points
        t_start = np.array([0])
        f_start = np.array([0])
        t_end = np.array([T.max()])
        f_end = np.array([f[-1]])
        
        # Get estimate of cum hazard function
        t = np.concatenate((t_start, t, t_end), axis=0)
        f = np.concatenate((f_start, f, f_end), axis=0)

        return t, f
        
    #%%===========================================================================
    # Actual prediction model
    #==============================================================================

    def predict(self, neighbor_idxs,
                Survival_train, Censored_train, 
                Survival_test = None, Censored_test = None, 
                K = 30, Method = "cumulative-time"):
        
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
                
        elif Method in ['cumulative-time', 'cumulative-hazard']:

                # itirate through patients

                for idx in range(N_test):
                    
                    # Get time and censorship
                    T = Survival_train[neighbor_idxs[idx, :]]
                    C = Censored_train[neighbor_idxs[idx, :]]
    
                    if C.min() == 1:
                        # All cases are censored
                        if Method == "cumulative-time":
                            T_test[idx] = T.max()
                        elif Method == "cumulative-hazard":
                            T_test[idx] = 0
                        continue
                        
                    if Method == "cumulative-time":
                    
                        # Get km estimator
                        t, f = self._km_estimator(T, C)
                    
                        # Get mean survival time
                        T_test[idx] = np.sum(np.diff(t) * f[0:-1])
                    
                    elif Method == 'cumulative-hazard':
                    
                        # Get NA estimator
                        T = Survival_train[neighbor_idxs[idx, :]]
                        C = Censored_train[neighbor_idxs[idx, :]]
                        t, f = self._na_estimator(T, C)
                    
                        # Get integral under cum. hazard curve
                        T_test[idx] = np.sum(np.diff(t) * f[0:-1])
        
        else:
            raise ValueError("Method not implemented.")
                   
        
        # Get c-index
        CI = 0
        if Method == "cumulative-hazard":
            prediction_type = "risk"
        else:
            prediction_type = "survival_time"

        if Survival_test is not None:
            assert (Censored_test is not None)
            CI = sUtils.c_index(T_test, Survival_test, Censored_test, 
                                prediction_type= prediction_type)
            
        return T_test, CI



    #%%===========================================================================
    # model tuning
    #============================================================================== 

    def tune_k(self, X, Survival, Censored,
               kcv=4, shuffles=5, \
               Ks=list(np.arange(10, 160, 10)),\
               norm=2, Method = "cumulative-time"):

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
            neighbor_idxs = self._get_neighbor_idxs(X_test, X_train, norm = norm)
        
            
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

    def ensemble_feat_rank(self, X, T, C,
                           featnames=None, 
                           kcv=4, shuffles=5, 
                           n_ensembles=50,
                           subset_size=30,
                           K=30,
                           Method='cumulative-time',
                           norm=2):
        """
        Perform feature selection using random ensembles.
        Random ensembles of size subset_size are used to 
        predict survival using KNN method and the model
        accuracy is noted. Features are ranked by the median
        accuracy of ensembles in which they appear.
        
        Args:
        -----
        X, T, C - *optimization/validation* set
        kcv, shuffle - cross validation params
        n_ensembles, subset_size - ensemble params
        K, method, norm - KNN params
        """

        # Get split indices over optimization set
        splitIdxs = \
         dm.get_balanced_SplitIdxs(C,
                                   K=kcv, SHUFFLES=shuffles,
                                   USE_OPTIM=False)
        
        # Initialize accuracy
        n_folds = len(splitIdxs['fold_cv_train'][0])
        feat_ci = np.empty((n_ensembles, X.shape[1], n_folds))
        feat_ci[:] = np.nan
        
        # Itirate through folds
        
        for fold in range(n_folds):
        
            # Isolate indices
            train_idxs = splitIdxs['fold_cv_train'][0][fold]
            test_idxs = splitIdxs['fold_cv_test'][0][fold]
        
            # Generate random ensembles
            ensembles = np.random.randint(0, X.shape[1], [n_ensembles, subset_size])
            
            print("\n\tfold\tensemble\tCi")

            for eidx in range(n_ensembles):
            
                # get neighbor indices based on this feature ensemble
                fidx = ensembles[eidx, :]
                neighborIdxs = self._get_neighbor_idxs(\
                                X[test_idxs, :][:, fidx], 
                                X[train_idxs, :][:, fidx], 
                                norm=norm)
        
                # get accuracy
                _, ci = self.predict(\
                         neighborIdxs, 
                         T[train_idxs], 
                         C[train_idxs],
                         Survival_test=T[test_idxs], 
                         Censored_test=C[test_idxs],
                         K=K,
                         Method=Method)
        
                feat_ci[eidx, fidx, fold] = ci
        
                print("\t{}\t{}\t{}".format(fold, eidx, round(ci, 3)))
        
        # Get feature ranks
        
        # median ci across ensembles in each fold
        median_ci = np.nanmedian(feat_ci, axis=0)
        # median ci accross all folds
        median_ci = np.nanmedian(median_ci, axis=1)
        
        feats_sorted = np.flip(np.argsort(median_ci), axis=0)

        if featnames is not None:
            featnames_sorted = featnames[feats_sorted]

            # save ranked feature list
            savename = self.RESULTPATH + self.description + "featnames_ranked.txt"
            with open(savename, 'wb') as f:
                np.savetxt(f, featnames_ranked, fmt='%s', delimiter='\t')

        else:
            featnames_sorted = None

        return median_ci, feats_sorted, featnames_sorted

    #==========================================================================   

    def get_optimal_n_feats(self, X, T, C,
                            kcv=4,
                            shuffles=2,
                            n_feats_max=100,
                            K=30,
                            Method='cumulative-time',
                            norm=2):

        """
        Find the optimal number of features to use
        from the NCA-transformed dataset.
        Similar concept to using the first principal 
        components of PCA_transformed datasets. 
        
        IMPORTANT: this assumes that the transformed features
        have been sorted by absolute feature weight 
        (from largest to smallest).

        Args:
        -----
        X, T, C - optimization set
        """
        
        # Get split indices over optimization set
        splitIdxs = dm.get_balanced_SplitIdxs(C, \
                                              K=kcv, SHUFFLES=shuffles,\
                                              USE_OPTIM = False)
        
        # Initialize
        n_folds = len(splitIdxs['fold_cv_train'][0])
        n_feats_all = np.arange(1, n_feats_max, 2)
        CIs = np.zeros([n_folds, len(n_feats_all)])
        
        for fold in range(n_folds):
        
            print("\n\tFold {} of {}".format(fold, n_folds-1))
            
            # Isolate patients belonging to fold
        
            idxs_train = splitIdxs['fold_cv_train'][0][fold]
            idxs_test = splitIdxs['fold_cv_test'][0][fold]
            
            X_test = X[idxs_test, :]
            X_train = X[idxs_train, :]
            T_train = T[idxs_train]
            C_train = C[idxs_train]
            T_test = T[idxs_test]
            C_test = C[idxs_test]
        
            print("\tn_feats \t Ci")
        
            for fidx, n_feats in enumerate(n_feats_all):
        
                # Get neighbor indices
                neighbor_idxs = self._get_neighbor_idxs(\
                        X_test[:, 0:n_feats], 
                        X_train[:, 0:n_feats], 
                        norm=norm)
            
                # Predict testing set
                _, Ci = self.predict(neighbor_idxs,
                                     T_train, C_train, 
                                     Survival_test=T_test, 
                                     Censored_test=C_test, 
                                     K=K, Method=Method)
        
        
        
                CIs[fold, fidx] = Ci
                
                print("\t{} \t {}".format(n_feats, round(Ci, 3)))
        
        
        # Get optimal K
        CIs_median = np.median(CIs, axis=0)
        CI_optim = np.max(CIs_median)
        
        n_feats_optim = n_feats_all[np.argmax(CIs_median)]
        print("\nOptimal: n_feats = {}, Ci = {}\n".format(n_feats_optim, round(CI_optim, 3)))

        return CIs, n_feats_optim
        

    #%%===========================================================================
    # model accuracy
    #============================================================================== 

    def cv_accuracy(self, X, Survival, Censored, \
                    splitIdxs, outer_fold, \
                    tune_params, \
                    norm = 2, Method = 'cumulative-time'):

        """
        Find model accuracy using KCV (after ptimizing K)
        
        X - features (n,d)
        Survival - survival (n,)
        Censored - censorship (n,)
        splitIdxs - dict; indices of patients belonging to each fold
        outer_fold - fold index for optimization and non-optim. sets
        tune_params - dict; parameters to pass to tune_k method
        """

        # Initialize
        n_folds = len(splitIdxs['fold_cv_train'][0])
        CIs = np.zeros([n_folds])

        print("\nOptimizing K for this outer fold.")

        # find optimal K on validation set
        optimIdxs = splitIdxs['idx_optim'][outer_fold]

        _, K_optim = self.tune_k(X[optimIdxs, :], \
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
            neighbor_idxs = self._get_neighbor_idxs(X_test, X_train, norm = norm)
        
            # Predict testing set
            _, Ci = self.predict(neighbor_idxs,
                                 Survival_train, Censored_train, 
                                 Survival_test = Survival_test, 
                                 Censored_test = Censored_test, 
                                 K = K_optim, Method = Method)
            
            CIs[fold] = Ci
               
            print("{} \t {} \t {}".format(outer_fold, fold, round(Ci, 3)))


        return CIs, K_optim
