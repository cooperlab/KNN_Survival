import numpy as np
from scipy.io import savemat

#basePath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
basePath = "/home/mtageld/Desktop/KNN_Survival/"

sites = ["GBMLGG", "BRCA", "KIPAN"]
dtypes = ["Integ", "Gene"]

for site in sites:
    for dtype in dtypes:

        print(site + "\t" + dtype)
        
        dataPath = basePath + 'Data/SingleCancerDatasets/' + site + '/' + site + '_' + dtype + '_Preprocessed.mat'
        splitIdxPath = dataPath.split('.mat')[0] + '_splitIdxs.pkl'
        
        splitIdxs = np.load(splitIdxPath)
        
        savemat(splitIdxPath.split('.pkl')[0] + '.mat', splitIdxs)
