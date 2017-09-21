#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:12:57 2017

@author: mohamed

A group of supporting utilities.
"""

import numpy as np
from scipy.io import loadmat
#import matplotlib.pylab as plt

#%%============================================================================
# Custom survival utilities
#==============================================================================

def getAliveStatus(Survival, Censored, t_min = 0, t_max = 0, scale = 1):

    """ 
    This converts survival and censored data (where 1 = lost to follow-up)
    into alive-dead data by adding a time indicator variable 
    (where 1 = alive, 0 = dead, -1 = unknown survival status).
    Use scale to convert time scale (eg to convert from days to weeks
    set scale = 7)
    """
    
    # Get data in needed scale
    Survival = np.int32(np.floor(Survival / scale))
    
    if t_max == 0:
        t_max = np.max(Survival)
    
    # Initialize output matrix - rows are patients, cols are time points
    aliveStatus = np.ones([len(Survival), t_max - t_min +1])
    
    #idx = 0
    for idx in range(len(Survival)):
    
        if Censored[idx] == 0:
            # known death time
            aliveStatus[idx,Survival[idx]+1:] = 0 
        else:
            # lost to follow-up
            aliveStatus[idx,Survival[idx]+1:] = -1
            
    return np.int32(aliveStatus)

#%%============================================================================
# Tools from SurvivalNet:
#  https://github.com/CancerDataScience/SurvivalNet/blob/master/survivalnet/ ...
#  optimization/SurvivalAnalysis.py
#==============================================================================

def c_index(prediction, T, C, prediction_type = 'risk'):
    
    """
    Calculate concordance index to evaluate model prediction.
    C-index calulates the fraction of all pairs of subjects whose predicted
    survival times are correctly ordered among all subjects that can actually
		be ordered, i.e. both of them are uncensored or the uncensored time of
		one is smaller than the censored survival time of the other.
    
    Parameters
    ----------
    prediction: numpy.ndarray
       m sized array of predicted risk/survival time
    T: numpy.ndarray
       m sized vector of time of death or last follow up
    C: numpy.ndarray
       m sized vector of censored status (do not confuse with observed status)
    prediction_type: either 'risk' or 'survival_time'
    Returns
    -------
    A value between 0 and 1 indicating concordance index. 
    """
    
    if prediction_type == 'risk':
        risk = prediction
        prediction = None
        
    elif prediction_type == 'survival_time':
        # normalize
        prediction = prediction / np.max(prediction)
        # convert to risk
        risk = 1 - prediction
        prediction  = None
    else:
        raise ValueError("prediction_type is either 'risk' or 'survival_time'.")
    
    # initialize    
    n_orderable = 0.0
    score = 0.0
    
    for i in range(len(T)):
        for j in range(i+1,len(T)):


            # Case 1: both cases are observed
            # =================================================================

            if(C[i] == 0 and C[j] == 0):

                # i and j are always orderable
                n_orderable = n_orderable + 1
                
                if(T[i] > T[j]):
                    if(risk[j] > risk[i]):
                        score = score + 1
                elif(T[j] > T[i]):
                    if(risk[i] > risk[j]):
                        score = score + 1
                else:
                    if(risk[i] == risk[j]):
                        score = score + 1

            # Case 2: i is censored while j is observed
            # =================================================================

            elif(C[i] == 1 and C[j] == 0):

                if(T[i] >= T[j]):      
                    
                    # i and j only orderable if i lived longer
                    n_orderable = n_orderable + 1
                    
                    if(T[i] > T[j]):
                        if(risk[j] > risk[i]):
                            score = score + 1

            # Case 3: i is observed while j is censored
            # =================================================================

            elif(C[j] == 1 and C[i] == 0):

                if(T[j] >= T[i]):
                    
                    # i and j only orderable if i lived longer
                    n_orderable = n_orderable + 1

                    if(T[j] > T[i]):
                        if(risk[i] > risk[j]):
                            score = score + 1
    
    if n_orderable > 0:
        ci = score / n_orderable
    else:
        ci = 0
    
    return ci

#==============================================================================

def calc_at_risk(T, O, X = None):
    
    """
    Calculate the at risk group of all patients.
		
	For every patient i, this function returns the index of the first 
	patient who died after i, after sorting the patients w.r.t. time of death.
    Refer to the definition of Cox proportional hazards log likelihood for
		details: https://goo.gl/k4TsEM
    
    Parameters
    ----------
    T: numpy.ndarray
       m sized vector of time of death
    O: numpy.ndarray
       m sized vector of observed status (1 - censoring status)
    X: (optional) numpy.ndarray
       m*n matrix of input data
    Returns
    -------
    T: numpy.ndarray
       m sized sorted vector of time of death
    O: numpy.ndarray
       m sized vector of observed status sorted w.r.t time of death
    at_risk: numpy.ndarray
       m sized vector of starting index of risk groups
    X: numpy.ndarray
       m*n matrix of input data sorted w.r.t time of death
    """
    tmp = list(T)
    T = np.asarray(tmp).astype('float64')
    order = np.argsort(T)
    sorted_T = T[order]
    at_risk = np.asarray([list(sorted_T).index(x) for x in sorted_T]).astype('int32')
    T = np.asarray(sorted_T)
    O = O[order]
    
    if X is not None:
        X = X[order, :]

    return T, O, at_risk, X


#%%############################################################################
#%%############################################################################
#%%############################################################################
#%%############################################################################

#%%============================================================================
# test methods
#==============================================================================

if __name__ == '__main__':
    
    # Load data
    dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
    Data = loadmat(dpath)
    
    data = np.float32(Data['Integ_X'])
    if np.min(Data['Survival']) < 0:
        Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1
    
    Survival = np.int32(Data['Survival'])
    Censored = np.int32(Data['Censored'])
    
    # Generate survival status - discretized into months
    aliveStatus = getAliveStatus(Survival, Censored, scale = 30)
    
    
