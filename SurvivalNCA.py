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

import SurvivalUtils as sUtils
import nca_cost

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
    
    def __init__(self, data, aliveStatus, 
                 RESULTPATH, description="", 
                 LOADPATH = None,
                 OBJECTIVE = 'Mahalanobis', 
                 SIGMA = 1, LEARN_RATE = 0.01, 
                 MONITOR_STEP = 1, N_SUBSET = 25):
        
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
            
            # "KL-divergence" of "Mahalanobis"
            self.OBJECTIVE = OBJECTIVE
            
            # None or between [0, 1], 
            # the larger the more emphasis on farther neighbors
            self.SIGMA = SIGMA
            
            self.LEARN_RATE = LEARN_RATE
            self.MONITOR_STEP = MONITOR_STEP
            
            # no of patients chosen randomly each time point
            self.N_SUBSET = N_SUBSET
    
    
            # Setting things up
            #======================================================================
            
            # Get dims
            self.D = data.shape[1]
            
            # Initialize A to a scaling matrix
            epsilon = 1e-7
            #A = np.eye(D)
            self.A = np.zeros((self.D, self.D))
            np.fill_diagonal(self.A, 1./(data.max(axis=0) - data.min(axis=0) + epsilon))
            
            # Initialize other
            self.costs = []
            self.ranks = None
            self.epochs = 0


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
            'OBJECTIVE' : self.OBJECTIVE,
            'SIGMA' : self.SIGMA,
            'LEARN_RATE' : self.LEARN_RATE,
            'MONITOR_STEP' : self.MONITOR_STEP,
            'N_SUBSET' : self.N_SUBSET,
            'D' : self.D,
            'A' : self.A,
            }
        
        if len(self.costs) > 0:
            attribs['costs'] = self.costs
            
        if self.ranks is not None:
            attribs['ranks'] = self.ranks
        
        return attribs
    
    #%%============================================================================
    # Core model
    #==============================================================================
    
    def _survival_nca_cost(self, data, aliveStatus):
        
        """Gets cumulative cost and gradient for all time points"""
    
        # initialize cum_f and cum_gradf
        cum_f = 0
        cum_gradf = np.zeros(self.A.shape)
        
        # no of time points
        T = aliveStatus.shape[1]
        
        #t = 20
        for t in range(T):
        
            print("epoch {}: t = {} of {}".format(self.epochs, t, T-1))
                
            # Get patients with known survival status at time t
            Y = aliveStatus[:, t]
            keep = Y >= 0 
            # proceed only if there's enough known patients
            if np.sum(0 + keep) < self.N_SUBSET:
                print("skipping current t ...")
                continue 
            Y = Y[keep]
            X = data[keep, :]
            
            # keep a random subset of patients (for efficiency)
            keep = np.random.randint(0, X.shape[0], self.N_SUBSET)
            Y = Y[keep]
            X = X[keep, :]
            
            if self.OBJECTIVE == 'Mahalanobis':
                f, gradf = nca_cost.cost(self.A.T, X.T, Y, SIGMA=self.SIGMA)
            elif self.OBJECTIVE == 'KL-divergence':
                f, gradf = nca_cost.cost_g(self.A.T, X.T, Y, SIGMA=self.SIGMA)
                
            cum_f += f
            cum_gradf += gradf.T # sum of derivative is derivative of sum
    
        return [cum_f, cum_gradf]
    
    
    #==========================================================================
    
    def train(self, data, aliveStatus):
        
        """ learns feature matrix A to maximize objective function"""
         
        try: 
            while True:
                
                print("\n--------------------------------------------")
                print("---- EPOCH = " + str(self.epochs))
                print("--------------------------------------------\n")
                
                [cum_f, cum_gradf] = self._survival_nca_cost(data, aliveStatus)
                
                # update A
                self.A += self.LEARN_RATE * cum_gradf
        
                # update costs
                self.costs.append([self.epochs, cum_f])
                
                # monitor
                if (self.epochs % self.MONITOR_STEP == 0) and (self.epochs > 0):
                    cs = np.array(self.costs)
                    self._plotMonitor(arr= cs, title= "cost vs. epoch", 
                                      xlab= "epoch", ylab= "cost", 
                                      savename= self.RESULTPATH + 
                                      self.description + "cost.svg")
                
                self.epochs += 1
                
        except KeyboardInterrupt:
            
            print("\nFinished training model.")
            
            # Save learned diagnoal elements of A (feature weights)
            A_diag = {'A_diag': np.diag(self.A)}
            savemat(self.RESULTPATH + self.description + 'A_diag', A_diag)
            
            # Save model
            self.save()
    
    
    #==========================================================================
    
    def rankFeats(self, data):
        
        """rank features by how variant they are after the new transformation"""
        
        print("Ranking features by variance after transformation ...")
    
        def _getRanks(A):
            
            Ax = np.dot(data, A)
        
            fvars = np.std(Ax, 0).reshape(self.D, 1)
            fidx = np.arange(len(A)).reshape(self.D, 1)
            
            fvars = np.concatenate((fidx, fvars), 1)
            fvars = fvars[fvars[:,1].argsort()][::-1]
            
            fnames_ranked = fnames[np.int32(fvars[:,0])]
            
            return fvars, fnames_ranked
        
        self.fvars_init, _ = _getRanks(np.eye(self.D, self.D))
        self.fvars, self.ranks = _getRanks(self.A)
        
        # save ranked features
        ranks = {'ranks': self.ranks}
        savemat(self.RESULTPATH + self.description + 'ranks', ranks)
        
        # Save model
        self.save()
    
            
    
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
    
    # Generate survival status - discretized into months
    aliveStatus = sUtils.getAliveStatus(Survival, Censored, scale = 30)
    
    #============================================================================
    # train a survival NCA model
    #==============================================================================
    
    ncaParams = {
        'LOADPATH' : None, #"/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/tmp/GBMLGG_Integ_ModelAttributes.txt",
        'RESULTPATH' : "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/tmp/",
        'description' : "BRCA_Integ_",
        'OBJECTIVE' : 'Mahalanobis',
        'SIGMA' : 1,
        'LEARN_RATE' : 0.01,
        'MONITOR_STEP' : 1,
        'N_SUBSET' : 20,
        }
    
    # instantiate model
    model = SurvivalNCA(data, aliveStatus, **ncaParams)
    modelInfo = model.getModelInfo()
    
    #raise(Exception)
    
    # train model and rank features
    model.train(data, aliveStatus)
    model.rankFeats(data)
    
    # inspect trained model
    modelInfo = model.getModelInfo()
    
    # some visualizations
    model.plot_stdChange()
    model.plot_scatterFeats(data, Survival, Censored, fidx1=10, fidx2=11)
    
    
    
    