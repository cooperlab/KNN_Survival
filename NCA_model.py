#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 01:56:26 2017

@author: mohamed

Survival NCA (Neighborhood Component Analysis)
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
from scipy.io import loadmat, savemat
from matplotlib import cm
import matplotlib.pylab as plt

import ProjectUtils as pUtils
import SurvivalUtils as sUtils
import DataManagement as dm
import NCA_graph as graph

raise(Exception)

#%%============================================================================
# NCAmodel class (trainable model)
#==============================================================================

class SurvivalNCA(object):
    
    """
    Extension of NCA to right-censored settings.
    
    Key references: 
        
        1- J Goldberger, GE Hinton, ST Roweis, RR Salakhutdinov. 
        Neighbourhood components analysis. 
        Advances in neural information processing systems, 513-520
        
        2- Yang, W., K. Wang, W. Zuo. 
        Neighborhood Component Feature Selection for High-Dimensional Data.
        Journal of Computers. Vol. 7, Number 1, January, 2012.
        
    """
    
    # Set class attributes
    ###########################################################################
    
    # default graph params
    default_graphParams = {'ALPHA': 0.1,
                           'LAMBDA': 1.0, 
                           'OPTIM': 'Adam',
                           'LEARN_RATE': 0.01}
    userspecified_graphParams = []
    
    
    # Init
    ###########################################################################
    
    def __init__(self, features, survival, censored, 
                 RESULTPATH, description="", 
                 LOADPATH = None,
                 COMPUT_GRAPH_PARAMS = {}):
        
        """Instantiate a survival NCA object"""
        
        if LOADPATH is not None:
            
            # Load existing model
            self.load(LOADPATH)
            
            # overwrite loaded paths
            self.RESULTPATH = RESULTPATH
            
        else:
            
            # Set instance attributes
            #==========================================================================
            
            self.RESULTPATH = RESULTPATH
            
            # prefix to all saved results
            self.description = description
            
            # Get dims
            self.D = features.shape[1]
            
            # new model inital attributes
            self.Errors_epochLevel_train = []
            self.Errors_epochLevel_valid = []
            self.Errors_batchLevel_train = []
            self.Errors_batchLevel_valid = []
            self.BATCHES_RUN = 0
            self.EPOCHS_RUN = 0
            
            # Params for the computational graph
            self.COMPUT_GRAPH_PARAMS = \
                pUtils.Merge_dict_with_default(\
                        dict_given = COMPUT_GRAPH_PARAMS,
                        dict_default = self.default_graphParams,
                        keys_Needed = self.userspecified_graphParams)
            self.COMPUT_GRAPH_PARAMS['dim_input'] = self.D


    #%%===========================================================================
    # Miscellaneous methods
    #==============================================================================
    
    # The following load/save methods are inspired by:
        # https://stackoverflow.com/questions/2345151/
        # how-to-save-read-class-wholly-in-python
        
    def save(self):
        
        """save class as ModelAttributes.txt"""
        
        print("Saving model attributes and results...")
        with open(self.RESULTPATH + self.description + 'ModelAttributes.txt','wb') as file:
            file.write(_pickle.dumps(self.__dict__))
            file.close()
    
    #==========================================================================
    
    def load(self, LOADPATH):
        
        """try to load ModelAttributes.txt"""
        
        print("Loading model attributes ...")
        with open(LOADPATH,'rb') as file:
            dataPickle = file.read()
            file.close()
            self.__dict__ = _pickle.loads(dataPickle)
            
    #==========================================================================
    
    def getModelInfo(self):
        
        """ Returns relevant model attributes"""
        
        attribs = {
            'RESULTPATH' : self.RESULTPATH,
            'description' : self.description,
            'D' : self.D,
            'Errors_epochLevel_train': self.Errors_epochLevel_train,
            'Errors_epochLevel_valid': self.Errors_epochLevel_valid,
            'Errors_batchLevel_train': self.Errors_batchLevel_train,
            'Errors_batchLevel_valid': self.Errors_batchLevel_valid,
            'BATCHES_RUN': self.BATCHES_RUN,
            'EPOCHS_RUN': self.EPOCHS_RUN,
            'COMPUT_GRAPH_PARAMS': self.COMPUT_GRAPH_PARAMS,
            }
        
        return attribs
    
    #%%============================================================================
    # Core model
    #==============================================================================
    
    
    #%%============================================================================
    # Visualization methods
    #==============================================================================
    
    
    def _plotMonitor(self, arr, title, xlab, ylab, savename):
                        
        """ plots cost/other metric to monitor progress """
        
        print("Plotting " + title)
        
        fig, ax = plt.subplots() 
        ax.plot(arr[:,0], arr[:,1], 'b', linewidth=1.5, aa=False)
        plt.title(title, fontsize =16, fontweight ='bold')
        plt.xlabel(xlab)
        plt.ylabel(ylab) 
        plt.tight_layout()
        plt.savefig(savename)
        plt.close() 
        
        #==========================================================================
    
    def plot_stdChange(self):
        
        """ plot stdev change"""
        
        print("Plotting feature stdev after transformation")
        
        fidx = np.arange(len(self.A)).reshape(self.D, 1)
        
        fig, ax = plt.subplots()
        
        ax.plot(fidx, self.fvars[:,1], 'b', linewidth=1.5, aa=False)
        ax.plot(fidx, self.fvars_init[:,1], 'k--', linewidth=1.5, aa=False)
        
        plt.ylim(ymax = 1.5)
        
        plt.title("feature stdev after transformation", fontsize =16, fontweight ='bold')
        plt.xlabel("feature index")
        plt.ylabel("feature stdev - \nbefore (k--) and after (b-) transformation")
        plt.savefig(self.RESULTPATH + self.description + "fvars.svg")
        plt.close()
        
        #==========================================================================

    def plot_scatterFeats(self, data, Survival, Censored, 
                      fidx1 = 0, fidx2 = 1):
    
        """ 
        scatter patients by two features and code their survival.
        Note: for best visual results, at least one of the features should 
        be continuous.
        """
        
        print("Plotting Features (transformed) vs survival (color)")
        
        Ax = np.dot(data, self.A)
        
        fig, ax = plt.subplots()
        
        keep = (Censored == 0).reshape(data.shape[0])
        X1 = Ax[keep, int(self.fvars[fidx1,0])]
        X2 = Ax[keep, int(self.fvars[fidx2,0])]
        Ys = Survival[keep,:]
        
        colors = cm.seismic(np.linspace(0, 1, len(Ys)))
        
        ax.scatter(X1, X2, color=colors)
        plt.title("Features (transformed) vs survival (color)", 
                  fontsize =16, fontweight ='bold')
        plt.xlabel(str(self.ranks[fidx1]), fontsize=5)
        plt.ylabel(self.ranks[fidx2], fontsize=5)
        plt.savefig(self.RESULTPATH + self.description + "scatterFeats.svg")
        plt.close()
    
    
#%% ###########################################################################
#%%
#%% ###########################################################################
#%%
#%% ###########################################################################


if __name__ == '__main__':

    #============================================================================
    # Load and preprocess data
    #==============================================================================
    
    # Load data
    #dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
    dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Gene.mat"
    #dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat"
    
    Data = loadmat(dpath)
    
    #data = np.float32(Data['Integ_X'])
    data = np.float32(Data['Gene_X'])
    
    if np.min(Data['Survival']) < 0:
        Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1
    
    Survival = np.int32(Data['Survival'])
    Censored = np.int32(Data['Censored'])
    #fnames = Data['Integ_Symbs']
    fnames = Data['Gene_Symbs']
    
    # remove zero-variance features
    fvars = np.std(data, 0)
    keep = fvars > 0
    data = data[:, keep]
    fnames = fnames[keep] 
