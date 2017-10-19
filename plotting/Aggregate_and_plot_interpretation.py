# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:36:05 2017

@author: mohamed
"""

#%%============================================================================
# Ground work
#==============================================================================
base_path = '/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/'
#base_path = '/home/mtageld/Desktop/KNN_Survival/'

result_path = base_path + 'Results/10_10Oct2017/Integ/'

sites = ["GBMLGG", "KIPAN"]
dtypes = ["Integ", ] #"Gene"]
methods = ["cumulative-time_TrueNCA_FalsePCA", "non-cumulative_TrueNCA_FalsePCA"]

#%%============================================================================
# Get feature ranks
#==============================================================================

site = sites[0]
dtype = dtypes[0]
method = methods[0]
#for site in sites:
#for dtype in dtypes:
#for method in methods:

ranks_path = result_path + method + '/' + site + '_' + dtype + '_/nca/ranks/'
save_path = base_path + 'Results/tmp/' + site + '_' + dtype + '_' + method.split('_')[0]


