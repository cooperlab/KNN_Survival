#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:04:49 2017

@author: mtageld
"""

import os

junk_path = '/home/mtageld/Desktop/KNN_Survival/Codes/junk/'

junk = os.listdir(junk_path)
junk = [j for j in junk if '.' in j]

for j in junk:
    os.system('mv ' + junk_path+j + ' ' + junk_path+'0_'+j)