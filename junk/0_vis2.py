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

import _pickle
import numpy as np
#from matplotlib import cm
#import matplotlib.pylab as plt

#=======================================================

RESULTPATH = "/home/mtageld/Desktop/KNN_Survival/Results/0_17Sep2017/"

#=======================================================

# Determine path components
NCA = "no_NCA"
norms = ["1_norm",] # "2_norm"]
methods = ["cumulative", "non_cumulative"]

sites = ["GBMLGG", "BRCA", "KIPAN", "LUSC"]
dtypes = ["Integ", "Gene"]

#=======================================================


for site in sites:
    for dtype in dtypes:
        for norm in norms:
            for method in methods:
                
                # Load results file
                dpath = RESULTPATH + NCA + "/" + norm + "/" + method + "/" +\
                        site + "_" + dtype + "_/"
                
                fpath = dpath + site + "_" + dtype + "_Results.pkl"
                
                with open(fpath, "rb") as f:
                    results = _pickle.load(f)
                
                print("\n" + dpath.split(RESULTPATH)[1])
                print("------------------------")
                print("25%: {}".format(np.percentile(results['CIs'], 25)))
                print("50%: {}".format(np.percentile(results['CIs'], 50)))
                print("75%: {}".format(np.percentile(results['CIs'], 75)))
                
                
