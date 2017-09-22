#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.io import savemat

#import sys
#sys.path.append("/home/mtageld/Desktop/KNN_Survival/Codes/")
#from KNNSurvival import SurvivalKNN as knn
#import DataManagement as dm

# ===============================================================
# Load dataset
# ===============================================================

dpath = '/home/mtageld/Desktop/KNN_Survival/Data/SingleCancerDatasets/MM/'

# load data
data_gene = pd.read_pickle(dpath + 'Integrated_MA_Gene.df')
data_clinical = pd.read_pickle(dpath + 'MA_Gene_Clinical.df')

# keep relevant info
# note: this assumes the data in gene expression
# and clinical file correspond to each other and
# are sorted (which is true) - BUT, the 
# columns in the gene expression files belong to 
# patients while the opposite is true for the 
# clinical file

#
# Gene expression
#
sample_id = list(data_gene.columns)
data_gene = np.float32(data_gene.values.T)


#
# Clinical
#

#feats_clinical = [j for j in data_clinical.columns if (\
#                  (j in ['D_Age', 'D_Gender']) or \
#                  ('CYTO' in j))]
feats_clinical = [j for j in data_clinical.columns if \
                    j in ['D_Age', 'D_Gender']]
clinical = data_clinical[feats_clinical]

gender = clinical['D_Gender']
gender[gender == 'Male'] = 1
gender[gender == 'Female'] = 0
clinical['D_Gender'] = gender

high_risk = data_clinical['HR_FLAG']
high_risk[high_risk == 'TRUE'] = 1
high_risk[high_risk == 'FALSE'] = 0
high_risk[high_risk == 'CENSORED'] = np.nan

fnames_clinical = np.array(clinical.columns)
clinical = np.float32(clinical.values)


#
# Combine and z-score
#

Gene_X = np.concatenate((clinical, data_gene), axis=1)
Gene_X = (Gene_X - np.mean(Gene_X, axis=0)) / np.std(Gene_X, axis=0)

Data = {'sample_id': sample_id,
        'Gene_X': Gene_X,
        'Survival': np.int32(data_clinical['D_OS'].values),
        'Censored': np.int32(1 - data_clinical['D_OS_FLAG'].values),
        'T_PFS': np.int32(data_clinical['D_PFS'].values),
        'C_PFS': np.int32(1 - data_clinical['D_PFS_FLAG'].values),
        'high_risk': np.float32(high_risk),
        }

# save
savemat(dpath + 'MM_Gene', Data)

