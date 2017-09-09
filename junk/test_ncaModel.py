# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:54:33 2017

@author: mohamed
"""

import sys
sys.path.append('/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Codes')

import numpy as np

import NCA_model as nca

#%%

RESULTPATH = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/tmp/"

ncamodel = nca.SurvivalNCA(RESULTPATH)
ncamodel.run(features = np.random.rand(200, 30),
             survival = np.random.randint(0, 300, [200,]),
             censored = np.random.binomial(1, 0.3, [200,]),
             features_valid = np.random.rand(100, 30),
             survival_valid = np.random.randint(0, 250, [100,]),
             censored_valid = np.random.binomial(1, 0.3, [100,]))