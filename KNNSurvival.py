#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 00:16:36 2017

@author: mohamed

Predict survival using KNN
"""

#def KNN_Survival(X_test,X_train,Alive_train,K,Beta1,Filters,sigma,Lambda)

"""
 This predicts survival based on the labels of K-nearest neighbours 
 using weighted euclidian distance and non-cumulative survival probability.

 INPUTS: 
 --------
 IMPORTANT: features as rows, samples as columns
 WARNING: Input data has to be normalized to have similar scales
 
 X_test - testing sample features
 X_train - training sample features
 Alive_train - Alive dead status (+1 (alive) --> -1 (dead))
 K - number of nearest-neighbours to use
 Beta - shrinkage factor --> higher values indicate less important
        features
 Filters - method of emphasizing or demphasizing neighbours 
 sigma - sigma of gaussian filter (lower values result in more emphasis on 
                                   closer neighbours)
 Lambda -  the larger lambda, the less penalty on lack of common
           dimensions (when there are no NAN values and lamda = 1, 
           there's no dimension penlty)
"""

