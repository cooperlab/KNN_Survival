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
import tensorflow as tf
#from scipy.io import loadmat, savemat
#from matplotlib import cm
#import matplotlib.pylab as plt

import logging
import datetime

import ProjectUtils as pUtils
import SurvivalUtils as sUtils
import DataManagement as dm
import NCA_graph as cgraph

#raise(Exception)

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
    userspecified_graphParams = ['dim_input',]
    
    
    # Init
    ###########################################################################
    
    def __init__(self, 
                 RESULTPATH, description="", 
                 LOADPATH = None):
        
        """Instantiate a survival NCA object"""
        
        if LOADPATH is not None:
            
            # Load existing model
            self.load(LOADPATH)
            
            # overwrite loaded paths
            self.RESULTPATH = RESULTPATH
            
        else:
            
            # Set instance attributes
            #==================================================================
            
            self.RESULTPATH = RESULTPATH
            self.LOGPATH = self.RESULTPATH + "model/logs/"
            self.WEIGHTPATH = self.RESULTPATH + "model/weights/"
            
            # prefix to all saved results
            self.description = description
            
            # new model inital attributes
            self.Errors_epochLevel_train = []
            self.Errors_epochLevel_valid = []
            self.Errors_batchLevel_train = []
            self.Errors_batchLevel_valid = []
            self.BATCHES_RUN = 0
            self.EPOCHS_RUN = 0
            
            # Create output dirs
            #==================================================================
            
            self._makeSubdirs()
            
            # Configure logger - will not work with iPython
            #==================================================================
            
            timestamp = str(datetime.datetime.today()).replace(' ','_')
            logging.basicConfig(filename = self.LOGPATH + timestamp + "_RunLogs.log", 
                                level = logging.INFO,
                                format = '%(levelname)s:%(message)s')
                                
            
    #%%===========================================================================
    # Miscellaneous methods
    #==============================================================================
    
    # The following load/save methods are inspired by:
    # https://stackoverflow.com/questions/2345151/
    # how-to-save-read-class-wholly-in-python
        
    def save(self):
        
        """save relevant attributes as ModelAttributes.pkl"""
        
        pUtils.Log_and_print("Saving relevant attributes ...")

        attribs = self.getModelInfo()
                
        with open(self.RESULTPATH + self.description + 
                  'model/ModelAttributes.pkl','wb') as f:
            _pickle.dump(attribs, f)
    
    #==========================================================================
    
    def load(self, LOADPATH):
        
        """load ModelAttributes.pkl"""
        
        print("Loading model attributes ...")
        
        with open(LOADPATH,'rb') as f:
            attribs = _pickle.load(f)
            
        # unpack dict
        self.RESULTPATH = attribs['RESULTPATH']
        self.description = attribs['description']
        self.Errors_epochLevel_train = attribs['Errors_epochLevel_train']
        self.Errors_epochLevel_valid = attribs['Errors_epochLevel_valid']
        self.Errors_batchLevel_train = attribs['Errors_batchLevel_train']
        self.Errors_batchLevel_valid = attribs['Errors_batchLevel_valid']
        self.BATCHES_RUN = attribs['BATCHES_RUN']
        self.EPOCHS_RUN = attribs['EPOCHS_RUN']
        self.COMPUT_GRAPH_PARAMS = attribs['COMPUT_GRAPH_PARAMS']
        self.LOGPATH = attribs['LOGPATH']
        self.WEIGHTPATH = attribs['WEIGHTPATH']
            
    #==========================================================================
    
    def getModelInfo(self):
        
        """ Returns relevant model attributes"""
        
        attribs = {
            'RESULTPATH' : self.RESULTPATH,
            'description' : self.description,
            'Errors_epochLevel_train': self.Errors_epochLevel_train,
            'Errors_epochLevel_valid': self.Errors_epochLevel_valid,
            'Errors_batchLevel_train': self.Errors_batchLevel_train,
            'Errors_batchLevel_valid': self.Errors_batchLevel_valid,
            'BATCHES_RUN': self.BATCHES_RUN,
            'EPOCHS_RUN': self.EPOCHS_RUN,
            'COMPUT_GRAPH_PARAMS': self.COMPUT_GRAPH_PARAMS,
            'LOGPATH': self.LOGPATH,
            'WEIGHTPATH': self.WEIGHTPATH,
            }
        
        return attribs
    
    #==========================================================================
    
    def reset_TrainHistory(self):
        
        """Resets training history (errors etc)"""  
        
        self.EPOCHS_RUN = 0
        self.BATCHES_RUN = 0    
        self.Errors_batchLevel_train = []            
        self.Errors_batchLevel_valid = []
        self.Errors_epochLevel_train = []
        self.Errors_epochLevel_valid = []
        self.save()
        
    #==========================================================================    
    
    def _makeSubdirs(self):
        
        """ Create output directories"""
        
        # Create relevant result subdirectories
        pUtils.makeSubdir(self.RESULTPATH, 'plots')
        pUtils.makeSubdir(self.RESULTPATH, 'ranks')
        
        # Create a subdir to save the model
        pUtils.makeSubdir(self.RESULTPATH, 'model')
        pUtils.makeSubdir(self.RESULTPATH + 'model/', 'weights')
        pUtils.makeSubdir(self.RESULTPATH + 'model/', 'logs')
        
    
    #%%============================================================================
    # build computational graph
    #==============================================================================
    
    def _build_computational_graph(self, COMPUT_GRAPH_PARAMS={}):
        
        """ 
        Build the computational graph for this model
        At least, no of dimensions ('D') must be provided
        """
        
        # Now that the computationl graph is provided D is always fixed
        self.D = COMPUT_GRAPH_PARAMS['dim_input']

        # Params for the computational graph
        self.COMPUT_GRAPH_PARAMS = \
            pUtils.Merge_dict_with_default(\
                    dict_given = COMPUT_GRAPH_PARAMS,
                    dict_default = self.default_graphParams,
                    keys_Needed = self.userspecified_graphParams)
                    
        # instantiate computational graph
        graph = cgraph.comput_graph(**self.COMPUT_GRAPH_PARAMS)
        
        return graph
    
    
    #%%============================================================================
    #  Run session   
    #==============================================================================
        
    def train(self, 
            features, survival, censored,
            features_valid = None, 
            survival_valid = None, 
            censored_valid = None,
            COMPUT_GRAPH_PARAMS={},
            BATCH_SIZE = 20,
            MONITOR_STEP = 10):
                
        """
        train a survivalNCA model
        features - (N,D) np array
        survival and censored - (N,) np array
        """
        
        pUtils.Log_and_print("Training survival NCA model.")
        
        
        # Initial preprocessing and sanity checks
        #====================================================================== 
        
        pUtils.Log_and_print("Initial preprocessing.")
        
        assert len(features.shape) == 2
        assert len(survival.shape) == 1
        assert len(censored.shape) == 1
        
        if features_valid is not None:
            USE_VALID = True
            assert (features_valid.shape[1] == features.shape[1])
            assert (survival_valid is not None)
            assert (censored_valid is not None)
        
        #
        # Z-scoring survival (for numerical stability with optimizer)
        #
        
        # Combine training and validation (for comparability)
        survival_all = survival[:, None]
        if USE_VALID:
            survival_all = np.concatenate((survival_all, 
                                           survival_valid[:, None]), axis=0)

        # z-score combined
        survival_all = (survival_all - np.mean(survival_all)) / np.std(survival_all)

        # separate out
        survival = survival_all[0:len(survival), 0]
        if USE_VALID:        
            survival_valid = survival_all[len(survival):, 0]
        
        #
        # Getting at-risk for validation set
        #
        if USE_VALID:
            features_valid, survival_valid, observed_valid, at_risk_valid = \
                sUtils.calc_at_risk(features_valid, survival_valid, 1-censored_valid)       

        # Define computational graph
        #======================================================================        
        
        COMPUT_GRAPH_PARAMS['dim_input'] = features.shape[1]
        graph = self._build_computational_graph(COMPUT_GRAPH_PARAMS)
        
        
        # Begin session
        #======================================================================  

        pUtils.Log_and_print("Running TF session.")

        with tf.Session() as sess:
            
            
            # Initial ground work
            #==================================================================
            
            # op to save/restore all the variables
            saver = tf.train.Saver()
            
            if "checkpoint" in os.listdir(self.WEIGHTPATH):
                # load existing weights 
                pUtils.Log_and_print("Restoring saved model ...")                
                saver.restore(sess, self.WEIGHTPATH + "model.ckpt")
                pUtils.Log_and_print("Model restored.")                
                
            else:                
                # start a new model
                sess.run(tf.global_variables_initializer())
                
            # for tensorboard visualization
            train_writer = tf.summary.FileWriter(self.RESULTPATH + 'model/tensorboard', 
                                                 sess.graph)
    
            
            try: 
                while True:
                    
                    # Shuffle so that training batches differ every epoch
                    #==========================================================
                    
                    idxs = np.arange(features.shape[0]);
                    np.random.shuffle(idxs)
                    features = features[idxs, :]
                    survival = survival[idxs]
                    censored  = censored[idxs]
            
                    # Divide into balanced batches
                    #==========================================================
                    
                    # Getting at-risk groups (trainign set)
                    features, survival, observed, at_risk = \
                      sUtils.calc_at_risk(features, survival, 1-censored)
                      
                    # Get balanced batches
                    self.batchIdxs = dm.get_balanced_batches(observed, BATCH_SIZE = BATCH_SIZE)
                    if USE_VALID:
                        self.batchIdxs_valid = \
                            dm.get_balanced_batches(observed_valid, BATCH_SIZE = BATCH_SIZE)  
                            
                    # Graph inputs
                    #==========================================================
                            
                    feed_dict = {graph.X_input: features,
                                 graph.T: survival,
                                 graph.O: observed,
                                 graph.At_Risk: at_risk,
                                 }        
                    if USE_VALID:       
                        feed_dict_valid = {graph.X_input: features_valid,
                                           graph.T: survival_valid,
                                           graph.O: observed_valid,
                                           graph.At_Risk: at_risk_valid,
                                           }
    


#                    _, cost = sess.run([g.optimizer, g.cost], feed_dict = feed_dict)
#                    cost_valid = g.cost.eval(feed_dict = feed_dict_valid)
#    
#                    # Normalize cost for sample size
#                    cost = cost / N
#                    cost_valid = cost_valid / N_valid
#                    
#                    print("epoch {}, cost_train = {}, cost_valid = {}"\
#                            .format(epochs, cost, cost_valid))
#                            
#                    # update costs
#                    costs.append([epochs, cost])
#                    costs_valid.append([epochs, cost_valid])
#                    
#                    # monitor
#                    if (epochs % MONITOR_STEP == 0) and (epochs > 0):
#                        
#                        cs = np.array(costs)
#                        cs_valid = np.array(costs_valid)
#                        
#                        _plotMonitor(arr= cs, arr2= cs_valid[:,1],
#                                     title= "cost vs. epoch", 
#                                     xlab= "epoch", ylab= "cost", 
#                                     savename= RESULTPATH + 
#                                     description + "cost.png")
#                    
#                    epochs += 1
#                    
#            except KeyboardInterrupt:
#                
#                print("\nFinished training model.")
#                print("Obtaining final results.")
#                
#                #W, B, X_transformed = sess.run([g.W, g.B, g.X_transformed], 
#                #                               feed_dict = feed_dict_valid)
#                
#                W, X_transformed = sess.run([g.W, g.X_transformed], 
#                                             feed_dict = feed_dict_valid)
#                
#                # save learned weights
#                np.save(RESULTPATH + description + 'weights.npy', W)

                        


    
#    #%%============================================================================
#    # Visualization methods
#    #==============================================================================
#    
#    
#    def _plotMonitor(arr, title, xlab, ylab, savename, arr2 = None):
#                            
#        """ plots cost/other metric to monitor progress """
#        
#        pUtils.Log_and_print("Plotting " + title)
#        
#        fig, ax = plt.subplots() 
#        ax.plot(arr[:,0], arr[:,1], 'b', linewidth=1.5, aa=False)
#        if arr2 is not None:
#            ax.plot(arr[:,0], arr2, 'r', linewidth=1.5, aa=False)
#        plt.title(title, fontsize =16, fontweight ='bold')
#        plt.xlabel(xlab)
#        plt.ylabel(ylab) 
#        plt.tight_layout()
#        plt.savefig(savename)
#        plt.close() 
#        
#        #==========================================================================
#    
#    def plot_stdChange(self):
#        
#        """ plot stdev change"""
#        
#        pUtils.Log_and_print("Plotting feature stdev after transformation")
#        
#        fidx = np.arange(len(self.A)).reshape(self.D, 1)
#        
#        fig, ax = plt.subplots()
#        
#        ax.plot(fidx, self.fvars[:,1], 'b', linewidth=1.5, aa=False)
#        ax.plot(fidx, self.fvars_init[:,1], 'k--', linewidth=1.5, aa=False)
#        
#        plt.ylim(ymax = 1.5)
#        
#        plt.title("feature stdev after transformation", fontsize =16, fontweight ='bold')
#        plt.xlabel("feature index")
#        plt.ylabel("feature stdev - \nbefore (k--) and after (b-) transformation")
#        plt.savefig(self.RESULTPATH + self.description + "fvars.svg")
#        plt.close()
#        
#        #==========================================================================
#
#    def plot_scatterFeats(self, data, Survival, Censored, 
#                      fidx1 = 0, fidx2 = 1):
#    
#        """ 
#        scatter patients by two features and code their survival.
#        Note: for best visual results, at least one of the features should 
#        be continuous.
#        """
#        
#        pUtils.Log_and_print("Plotting Features (transformed) vs survival (color)")
#        
#        Ax = np.dot(data, self.A)
#        
#        fig, ax = plt.subplots()
#        
#        keep = (Censored == 0).reshape(data.shape[0])
#        X1 = Ax[keep, int(self.fvars[fidx1,0])]
#        X2 = Ax[keep, int(self.fvars[fidx2,0])]
#        Ys = Survival[keep,:]
#        
#        colors = cm.seismic(np.linspace(0, 1, len(Ys)))
#        
#        ax.scatter(X1, X2, color=colors)
#        plt.title("Features (transformed) vs survival (color)", 
#                  fontsize =16, fontweight ='bold')
#        plt.xlabel(str(self.ranks[fidx1]), fontsize=5)
#        plt.ylabel(self.ranks[fidx2], fontsize=5)
#        plt.savefig(self.RESULTPATH + self.description + "scatterFeats.svg")
#        plt.close()
    
    
#%% ###########################################################################
#%%
#%% ###########################################################################
#%%
#%% ###########################################################################

#
#if __name__ == '__main__':
#
#    #============================================================================
#    # Load and preprocess data
#    #==============================================================================
#    
#    # Load data
#    #dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
#    dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Gene.mat"
#    #dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat"
#    
#    Data = loadmat(dpath)
#    
#    #data = np.float32(Data['Integ_X'])
#    data = np.float32(Data['Gene_X'])
#    
#    if np.min(Data['Survival']) < 0:
#        Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1
#    
#    Survival = np.int32(Data['Survival'])
#    Censored = np.int32(Data['Censored'])
#    #fnames = Data['Integ_Symbs']
#    fnames = Data['Gene_Symbs']
#    
#    # remove zero-variance features
#    fvars = np.std(data, 0)
#    keep = fvars > 0
#    data = data[:, keep]
#    fnames = fnames[keep] 
