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
import matplotlib.pylab as plt

#import logging
import datetime

import ProjectUtils as pUtils
import SurvivalUtils as sUtils
import DataManagement as dm

#import NCA_graph as cgraph
import NCA_graph as cgraph
import KNNSurvival as knn

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
    default_graphParams = {'ALPHA': 0.5,
                           'LAMBDA': 0,
                           'SIGMA': 1.0,
                           'OPTIM': 'GD',
                           'LEARN_RATE': 0.01,
                           'per_split_feats': 500,
                           'ROTATE': False,
                           }
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
            self.Costs_epochLevel_train = []
            self.CIs_train = []
            self.CIs_valid = []
            self.BATCHES_RUN = 0
            self.EPOCHS_RUN = 0
            
            # Create output dirs
            #==================================================================
            
            self._makeSubdirs()
            
            # Configure logger - will not work with iPython
            #==================================================================
            
            #timestamp = str(datetime.datetime.today()).replace(' ','_')
            #logging.basicConfig(filename = self.LOGPATH + timestamp + "_RunLogs.log", 
            #                    level = logging.INFO,
            #                    format = '%(levelname)s:%(message)s')
                                
            
    #%%===========================================================================
    # Miscellaneous methods
    #==============================================================================
    
    # The following load/save methods are inspired by:
    # https://stackoverflow.com/questions/2345151/
    # how-to-save-read-class-wholly-in-python
        
    def save(self):
        
        """save relevant attributes as ModelAttributes.pkl"""
        
        #pUtils.Log_and_print("Saving relevant attributes ...")

        attribs = self.getModelInfo()
                
        with open(self.RESULTPATH + 'model/' + self.description + \
                  'ModelAttributes.pkl','wb') as f:
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
        self.Costs_epochLevel_train = attribs['Costs_epochLevel_train']
        self.CIs_train = attribs['CIs_train']
        self.CIs_valid = attribs['CIs_valid']        
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
            'Costs_epochLevel_train': self.Costs_epochLevel_train,
            'CIs_train': self.CIs_train,
            'CIs_valid': self.CIs_valid,
            'BATCHES_RUN': self.BATCHES_RUN,
            'EPOCHS_RUN': self.EPOCHS_RUN,
            'COMPUT_GRAPH_PARAMS': self.COMPUT_GRAPH_PARAMS,
            'LOGPATH': self.LOGPATH,
            'WEIGHTPATH': self.WEIGHTPATH,
            }
        
        return attribs
    
    #==========================================================================
    
    def reset_TrainHistory(self):
        
        """Resets training history (Costs etc)"""  
        
        self.EPOCHS_RUN = 0
        self.BATCHES_RUN = 0    
        self.Costs_epochLevel_train = []
        self.CIs_train = []
        self.CIs_valid = []
        #self.save()
        
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
            PLOT_STEP = 10,
            MODEL_SAVE_STEP = 10,
            MAX_ITIR = 100,
            MODEL_BUFFER = 4):
                
        """
        train a survivalNCA model
        features - (N,D) np array
        survival and censored - (N,) np array
        """
        
        #pUtils.Log_and_print("Training survival NCA model.")
        
        
        # Initial preprocessing and sanity checks
        #====================================================================== 
        
        #pUtils.Log_and_print("Initial preprocessing.")
        
        assert len(features.shape) == 2
        assert len(survival.shape) == 1
        assert len(censored.shape) == 1
        
        USE_VALID = False
        if features_valid is not None:
            USE_VALID = True
            assert (features_valid.shape[1] == features.shape[1])
            assert (survival_valid is not None)
            assert (censored_valid is not None)
        
        # Define computational graph
        #======================================================================        
        
        D = features.shape[1]
        COMPUT_GRAPH_PARAMS['dim_input'] = D
        graph = self._build_computational_graph(COMPUT_GRAPH_PARAMS)
        
        
        # Begin session
        #======================================================================  
        
        #print("Running TF session.")
        #pUtils.Log_and_print("Running TF session.")

        with tf.Session() as sess:
            
            
            # Initial ground work
            #==================================================================
            
            # op to save/restore all the variables
            saver = tf.train.Saver()
            
            if "checkpoint" in os.listdir(self.WEIGHTPATH):
                # load existing weights 
                #pUtils.Log_and_print("Restoring saved model ...")                
                saver.restore(sess, self.WEIGHTPATH + self.description + ".ckpt")
                #pUtils.Log_and_print("Model restored.")                
                
            else:                
                # start a new model
                sess.run(tf.global_variables_initializer())
                
            # for tensorboard visualization
            #train_writer = tf.summary.FileWriter(self.RESULTPATH + 'model/tensorboard', 
            #                                     sess.graph)

            # Define some methods
            #==================================================================


            # periodically save model
            def _saveTFmodel():

                """Saves model weights using tensorflow saver"""
            
                # save weights                        
                #pUtils.Log_and_print("\nSaving TF model weights...")
                #save_path = saver.save(sess, \
                #                self.WEIGHTPATH + self.description + ".ckpt")
                #pUtils.Log_and_print("Model saved in file: %s" % save_path)
            
                # save attributes
                self.save()
             
            
            # monitor
            def _monitorProgress(PLOT=True):

                """
                Monitor cost - save txt and plots cost
                """
                
                # find min epochs to display in case of keyboard interrupt
                max_epoch = np.min([len(self.Costs_epochLevel_train),
                                    len(self.CIs_train),
                                    len(self.CIs_valid)])
                
                # concatenate costs
                costs = np.array(self.Costs_epochLevel_train[0:max_epoch])
                cis_train = np.array(self.CIs_train[0:max_epoch])
                if USE_VALID:
                    cis_valid = np.array(self.CIs_valid[0:max_epoch])
                else:
                    cis_valid = None
                
                epoch_no = np.arange(max_epoch)
                costs = np.concatenate((epoch_no[:, None], costs), axis=1)
                cis_train = np.concatenate((epoch_no[:, None], cis_train), axis=1)
                
                # Get unique datetime identifier
                timestamp = str(datetime.datetime.today()).replace(' ','_')
                timestamp.replace(":", '_')
                
                # Saving raw numbers for later reference
                savename= self.RESULTPATH + "plots/" + self.description + "cost_" + timestamp
                
                with open(savename + '_costs.txt', 'wb') as f:
                    np.savetxt(f, costs, fmt='%s', delimiter='\t')

                with open(savename + '_cis_train.txt', 'wb') as f:
                    np.savetxt(f, cis_train, fmt='%s', delimiter='\t')

                if USE_VALID:
                    with open(savename + '_cis_valid.txt', 'wb') as f:
                        np.savetxt(f, cis_valid, fmt='%s', delimiter='\t')
                
                #
                # Note, plotting would not work when running
                # this using screen (Xdisplay is not supported)
                #
                if PLOT:
                    self._plotMonitor(arr= cis_train,
                                      title= "Cost vs. epoch", 
                                      xlab= "epoch", ylab= "Cost", 
                                      savename= savename + ".svg")
                    self._plotMonitor(arr= cis_train, arr2= cis_valid,
                                      title= "C-index vs. epoch", 
                                      xlab= "epoch", ylab= "C-index", 
                                      savename= savename + ".svg")

    
            # Begin epochs
            #==================================================================
            
            try: 
                itir = 0
                
                print("\n\tepoch\tcost\tCi_train\tCi_valid")
                print("\t----------------------------------------------")
                
                knnmodel = knn.SurvivalKNN(self.RESULTPATH, description=self.description)
                
                # Initialize weights buffer 
                # (keep a snapshot of model for early stopping)
                # each "channel" in 3rd dim is one snapshot of the model
                if USE_VALID:
                    Ws = np.zeros((D, D, MODEL_BUFFER))
                    Cis = []
                
                while itir < MAX_ITIR:
                    
                    #pUtils.Log_and_print("\n\tTraining epoch {}\n".format(self.EPOCHS_RUN))
                    
                    itir += 1
                    cost_tot = 0
            
                    # Divide into balanced batches
                    #==========================================================
                    
                    n = censored.shape[0]
                    
                    # Get balanced batches (if relevant)
                    if BATCH_SIZE < n:
                        # Shuffle so that training batches differ every epoch
                        idxs = np.arange(features.shape[0]);
                        np.random.shuffle(idxs)
                        features = features[idxs, :]
                        survival = survival[idxs]
                        censored  = censored[idxs]
                        # stochastic mini-batch GD
                        batchIdxs = dm.get_balanced_batches(censored, 
                                                            BATCH_SIZE=BATCH_SIZE)
                    else:
                        # Global GD
                        batchIdxs = [np.arange(n)]
                                      
                    # Run over training set
                    #==========================================================
                            
                    for batchidx, batch in enumerate(batchIdxs):
                        
                        # Getting at-risk groups
                        t_batch, o_batch, at_risk_batch, x_batch = \
                            sUtils.calc_at_risk(survival[batch], 
                                                1-censored[batch],
                                                features[batch, :])
                                                
                        # Get at-risk mask (to be multiplied by Pij)
                        Pij_mask = np.zeros((n, n))
                        for idx in range(n):
                            # only observed cases
                            if o_batch[idx] == 1:
                                # only at-risk cases
                                Pij_mask[idx, at_risk_batch[idx]:] = 1
                        
                        # run optimizer and fetch cost
                        feed_dict = {graph.X_input: x_batch,
                                     graph.Pij_mask: Pij_mask,
                                     }  
                        _, cost, W = sess.run([graph.optimizer, 
                                               graph.cost, 
                                               graph.W], \
                                               feed_dict = feed_dict)
                                                 
                        # normalize cost for sample size
                        cost = cost / len(batch)
                        
                        # record/append cost
                        #self.Costs_batchLevel_train.append(cost)                  
                        cost_tot += cost                        

                        #pUtils.Log_and_print("\t\tTraining: Batch {} of {}, cost = {}".\
                        #     format(batchidx, len(batchIdxs)-1, round(cost[0], 3)))
                    
                    
                    # Get Ci for training/validation set
                    #==========================================================
            
                    # transform
                    x_train_transformed = np.dot(x_batch, W)
                    if USE_VALID:
                        x_valid_transformed = np.dot(features_valid, W)
            
                    # get neighbor indices    
                    neighbor_idxs_train = \
                        knnmodel._get_neighbor_idxs(x_train_transformed, 
                                                    x_train_transformed, 
                                                    norm = 2)
                    if USE_VALID:
                        neighbor_idxs_valid = \
                            knnmodel._get_neighbor_idxs(x_valid_transformed, 
                                                        x_train_transformed, 
                                                        norm = 2)
                    
                    # Predict training/validation set
                    _, Ci_train = knnmodel.predict(neighbor_idxs_train,
                                                   Survival_train=t_batch, 
                                                   Censored_train=1-o_batch, 
                                                   Survival_test=t_batch, 
                                                   Censored_test=1-o_batch, 
                                                   K = 30, 
                                                   Method = "cumulative-time")
                    if USE_VALID:
                        _, Ci_valid = knnmodel.predict(neighbor_idxs_valid,
                                                   Survival_train=t_batch, 
                                                   Censored_train=1-o_batch, 
                                                   Survival_test=survival_valid, 
                                                   Censored_test=censored_valid, 
                                                   K = 30, 
                                                   Method = "cumulative-time")
                    if not USE_VALID:
                        Ci_valid = 0
                    print("\t{}\t{}\t{}\t{}".format(\
                            self.EPOCHS_RUN,
                            round(cost_tot, 3), 
                            round(Ci_train, 3),
                            round(Ci_valid, 3)))                                                   
                                                   
                    # Early stopping
                    #==========================================================

                    if USE_VALID:                                                   
                        # Save snapshot                        
                        Ws[:, :, itir % MODEL_BUFFER] = W 
                        Cis.append(Ci_valid)
                        
                        # Stop when overfitting starts to occur
                        if len(Cis) > (2 * MODEL_BUFFER):
                            ci_new = np.mean(Cis[-MODEL_BUFFER:])
                            ci_old = np.mean(Cis[-2*MODEL_BUFFER:-MODEL_BUFFER])        
                    
                            if ci_new < ci_old:
                                snapshot_idx = (itir - MODEL_BUFFER) % MODEL_BUFFER
                                W = Ws[:, :, snapshot_idx]
                                break
                    
                    
                    # Update and save                     
                    #==========================================================
                    
                    # update epochs and append costs                     
                    self.EPOCHS_RUN += 1
                    self.Costs_epochLevel_train.append(cost_tot)
                    self.CIs_train.append(Ci_train)
                    self.CIs_valid.append(Ci_valid)
                   
                    # periodically save model
                    #if (self.EPOCHS_RUN % MODEL_SAVE_STEP) == 0:
                    #    _saveTFmodel() 
                    
                    # periodically monitor progress
                    if (self.EPOCHS_RUN % PLOT_STEP == 0) and \
                        (self.EPOCHS_RUN > 0):
                        _monitorProgress() 
                    
            except KeyboardInterrupt:
                pass
                
            # save final model and plot costs
            #_saveTFmodel()
            _monitorProgress()

            #pUtils.Log_and_print("Finished training model.")
            #pUtils.Log_and_print("Obtaining final results.")
            
            # save learned weights
            #W = sess.run(graph.W, feed_dict = feed_dict)
            np.save(self.RESULTPATH + 'model/' + self.description + \
                    'featWeights.npy', W)
            
        return W

                        
    #%%============================================================================
    # Rank features
    #==============================================================================

        
    def rankFeats(self, X, fnames, rank_type = "weights"):
        
        """ ranks features by feature weights or variance after transform"""
        
        print("Ranking features by " + rank_type)
    
        fidx = np.arange(self.D).reshape(self.D, 1)        
        
        w = np.load(self.RESULTPATH + 'model/' + self.description + 'featWeights.npy')        
        
        if rank_type == 'weights':
            # rank by feature weight
            ranking_metric = w[:, None]
        elif rank_type == 'stdev':
            # rank by variance after transform
            W = np.zeros([self.D, self.D])
            np.fill_diagonal(W, w)
            X = np.dot(X, W)
            ranking_metric = np.std(X, 0).reshape(self.D, 1)
        
        ranking_metric = np.concatenate((fidx, ranking_metric), 1)      
    
        # Plot feature weights/variance
        if self.D <= 500:
            n_plot = ranking_metric.shape[0]
        else:
            n_plot = 500
        self._plotMonitor(ranking_metric[0:n_plot,:], 
                          "feature " + rank_type, 
                          "feature_index", rank_type, 
                          self.RESULTPATH + "plots/" + self.description + 
                          "feat_" + rank_type+"_.svg")
        
        # rank features
        
        if rank_type == "weights":
            # sort by absolute weight but keep sign
            ranking = ranking_metric[np.abs(ranking_metric[:,1]).argsort()][::-1]
        elif rank_type == 'stdev':    
            ranking = ranking_metric[ranking_metric[:,1].argsort()][::-1]
        
        fnames_ranked = fnames[np.int32(ranking[:,0])].reshape(self.D, 1)
        fw = ranking[:,1].reshape(self.D, 1) 
        fnames_ranked = np.concatenate((fnames_ranked, fw), 1)
        
        # save results
        
        savename = self.RESULTPATH + "ranks/" + self.description +\
                    rank_type + "_ranked.txt"
        with open(savename,'wb') as f:
            np.savetxt(f,fnames_ranked,fmt='%s', delimiter='\t')

    
    #%%============================================================================
    # Visualization methods
    #==============================================================================
    
    def _plotMonitor(self, arr, title, xlab, ylab, savename, arr2 = None):
                            
        """ plots cost/other metric to monitor progress """
        
        print("Plotting " + title)
        
        fig, ax = plt.subplots() 
        ax.plot(arr[:,0], arr[:,1], 'b', linewidth=1.5, aa=False)
        if arr2 is not None:
            ax.plot(arr[:,0], arr2, 'r', linewidth=1.5, aa=False)
        plt.title(title, fontsize =16, fontweight ='bold')
        plt.xlabel(xlab)
        plt.ylabel(ylab) 
        plt.tight_layout()
        plt.savefig(savename)
        plt.close()

    #==========================================================================    
        
#%% ###########################################################################
#%% ###########################################################################
#%% ###########################################################################
#%% ###########################################################################
