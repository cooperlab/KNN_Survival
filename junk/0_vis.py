#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:58:18 2017

@author: mtageld
"""

import os
#import scipy.misc as sm
#import matplotlib.pylab as plt
from IPython.core.display import Image, display

#%%

#RESPATH = '/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/tmp/'
RESPATH = '/home/mtageld/Desktop/KNN_Survival/Results/tmp/'

ext = '.png'

fileList = os.listdir(RESPATH)
fileList = [j for j in fileList if ext in j]

for i in range(len(fileList)):
    display(Image(RESPATH + fileList[i]))
