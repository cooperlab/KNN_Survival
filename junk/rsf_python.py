import numpy as np

basePath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
site = "GBMLGG"
dtype = "Integ"
dataPath = basePath + 'Data/SingleCancerDatasets/' + site + '/' + site + '_' + dtype + '_Preprocessed.mat'
splitIdxPath = dataPath.split('.mat')[0] + '_splitIdxs.pkl'

splitIdxs = np.load(splitIdxPath)
