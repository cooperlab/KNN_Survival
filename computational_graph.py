#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 12:57:55 2017

@author: mohamed
"""

import tensorflow as tf

#%%============================================================================
# Computational graph class
#==============================================================================

class comput_graph(object):
    
    """
    Builds the computational graph for Survival NCA.
    """
    
    def __init__(self, dim_input, 
                 ALPHA = 0.5,
                 LAMBDA = 1,
                 OPTIM = 'GD',
                 LEARN_RATE = 0.01,
                 PIJ_LOOP = False):
        
        """
        Instantiate a computational graph for survival NCA.
        
        Args:
        ------
        dim_input - no of features
        
        """
        
        print("Building computational graph for survival NCA.")        
        
        # set up instace attributes
        self.dim_input = dim_input
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.OPTIM = OPTIM
        self.LEARN_RATE = LEARN_RATE
        self.PIJ_LOOP = PIJ_LOOP
        
        # clear lurking tensors
        tf.reset_default_graph()
        
        print("Adding placeholders.")
        self.add_placeholders()
        
        print("Adding linear feature transform.")
        self.add_linear_transform()
            
        print("Adding regularized weighted log likelihood.")
        self.add_cost()
        
        print("Adding optimizer.")
        self.add_optimizer()


    #%%========================================================================
    # Add placeholders to graph  
    #==========================================================================
    
    def add_placeholders(self):
    
        """ Adds graph inputs as placeholders in graph """
        
        with tf.variable_scope("Inputs"):
        
            self.X_input = tf.placeholder("float", [None, self.dim_input], name='X_input')
            
            self.T = tf.placeholder("float", [None], name='T')
            self.O = tf.placeholder("float", [None], name='O')
            self.At_Risk = tf.placeholder("float", [None], name='At_Risk')
            
            # type conversions
            self.T = tf.cast(self.T, tf.float32)
            self.O = tf.cast(self.O, tf.int32)
            self.At_Risk = tf.cast(self.At_Risk, tf.int32)
            
                 
    #%%========================================================================
    # Linear transformation (for interpretability)
    #==========================================================================

    def add_linear_transform(self):
        
        """ 
        Transform features in a linear fashion for better interpretability
        """
        
        with tf.variable_scope("linear_transform"):
            
            # feature scales/weights
            w = tf.get_variable("weights", shape=[self.dim_input], 
                            initializer= tf.contrib.layers.xavier_initializer())
            self.B = tf.get_variable("biases", shape=[self.dim_input], 
                            initializer= tf.contrib.layers.xavier_initializer())
            
            # diagonalize and matmul
            self.W = tf.diag(w)
            #self.W = tf.get_variable("weights", shape=[self.dim_input, self.dim_input], 
            #                initializer= tf.contrib.layers.xavier_initializer())
                        
            self.X_transformed = tf.add(tf.matmul(self.X_input, self.W), self.B) 
    
    #%%========================================================================
    # Get Pij 
    #==========================================================================

    def _get_Pij(self):
        
        """ 
        Calculate Pij, the probability that j will be chosen 
        as i's neighbor, for all i's
        """
        
        def get_pij_using_loop():
            
            """
            Gets Pij by looping through patients. Is appropriate when the 
            non-loop version causes memory errors.            
            """
            
            with tf.name_scope("pij_loop"):
                
                n = tf.shape(self.X_transformed)[0]
                n = tf.cast(n, tf.int32)
                #n = self.X_transformed.get_shape().as_list()[0]
                #n = tf.cast(tf.size(self.T)-1, tf.int32)
                
                # first patient
                #sID = 0
                #patient = self.X_transformed[sID, :]
                #patient_normax = (patient[None, :] - self.X_transformed)**2
                #normAX = tf.reduce_sum(patient_normax, axis=1)
                #normAX = normAX[None, :]
                normAX = tf.Variable(tf.zeros([n, n]), validate_shape=False)
                
                # all other patients
                def _append_normAX(sID, normAX):
                
                    """append normAX for a single patient to existing normAX"""
                    
                    # calulate normAX for this patient    
                    patient = self.X_transformed[sID, :]
                    patient_normax = (patient[None, :] - self.X_transformed)**2
                    patient_normax = tf.reduce_sum(patient_normax, axis=1)
                    patient_normax = patient_normax[None, :]
                
                    # append to existing list
                    #normAX = tf.concat((normAX, patient_normax[None, :]), axis=0)
                    #a = normAX[sID, :]
                    #normAX = tf.assign(normAX[sID, :], patient_normax[None, :])
                    normAX = normAX[sID, :].assign(patient_normax[None, :])
                    
                    # sID++
                    sID = tf.cast(tf.add(sID, 1), tf.int32)
                    
                    return sID, normAX
                    
                
                
                
                # Go through all patients and add normAX
                #sID = tf.cast(tf.Variable(1), tf.int32)
                sID = tf.cast(tf.Variable(0), tf.int32)
                
                c = lambda sID, normAX: tf.less(sID, tf.cast(n, tf.int32))
                b = lambda sID, normAX: _append_normAX(sID, normAX)
                
                (sID, normAX) = tf.while_loop(c, b, 
                                loop_vars = [sID, normAX])#, 
                                #shape_invariants = 
                                #[sID.get_shape(), tf.TensorShape([None, n])])
                            
            return normAX
        
        
        with tf.name_scope("getting_Pij"):
            
            if self.PIJ_LOOP:
                # use if running into memory errors
                normAX = get_pij_using_loop()
                
            else:
                #
                # Inspired by: https://github.com/RolT/NCA-python
                #
                
                # Expand dims of AX to [n_samples, n_samples, n_features], where
                # each "channel" in the third dimension is the difference between
                # one sample and all other samples along one feature
                normAX = self.X_transformed[None, :, :] - \
                         self.X_transformed[:, None, :]
                
                # Now get the euclidian distance between
                # every patient and all others -> [n_samples, n_samples]
                #normAX = tf.norm(normAX, axis=0)
                normAX = tf.reduce_sum(normAX ** 2, axis=2)
            
            # Calculate Pij, the probability that j will be chosen 
            # as i's neighbor, for all i's. Pij has shape
            # [n_samples, n_samples] and ** is NOT symmetrical **.
            # Because the data is normalized using softmax, values
            # add to 1 in rows, that is i (central patients) are
            # represented in rows
            denomSum = tf.reduce_sum(tf.exp(-normAX), axis=0)
            epsilon = 1e-8
            denomSum = denomSum + epsilon            
            
            self.Pij = tf.exp(-normAX) / denomSum[:, None]
    

    #%%========================================================================
    #  Loss function - weighted log likelihood   
    #==========================================================================

    def add_cost(self):
        
        """
        Adds penalized weighted likelihood to computational graph        
        """
    
        # Get Pij, probability j will be i's neighbor
        self._get_Pij()
        
        def _add_to_cumSum(Idx, cumsum):
        
            """Add patient to log partial likelihood sum """
            
            # Get survival of current patient and corresponding at-risk cases
            # i.e. those with higher survival or last follow-up time
            Pred_thisPatient = self.T[Idx]
            Pred_atRisk = self.T[self.At_Risk[Idx]:tf.size(self.T)-1]
            
            # Get Pij of at-risk cases from this patient's perspective
            Pij_thisPatient = self.Pij[Idx, self.At_Risk[Idx]:tf.size(self.T)-1]
            
            # exponentiate and weigh Pred_AtRisk
            Pred_atRisk = tf.multiply(tf.exp(Pred_atRisk), Pij_thisPatient)
            
            # Get log partial sum of prediction for those at risk
            LogPartialSum = tf.log(tf.reduce_sum(Pred_atRisk))
            
            # Get difference
            Diff_ThisPatient = tf.subtract(Pred_thisPatient, LogPartialSum)
            
            # Add to cumulative log partial likeliood sum
            cumsum = tf.add(cumsum, Diff_ThisPatient)
            
            return cumsum
    
        def _add_if_observed(Idx, cumsum):
        
            """ Add to cumsum if current patient'd death time is observed """
            
            with tf.name_scope("add_if_observed"):
                cumsum = tf.cond(tf.equal(self.O[Idx], 1), 
                                lambda: _add_to_cumSum(Idx, cumsum),
                                lambda: tf.cast(cumsum, tf.float32))                                    
    
                Idx = tf.cast(tf.add(Idx, 1), tf.int32)
            
            return Idx, cumsum
            
        def _penalty(W):
    
            """
            Elastic net penalty. Inspired by: 
            https://github.com/glm-tools/pyglmnet/blob/master/pyglmnet/pyglmnet.py
            """
            
            with tf.name_scope("Elastic_net"):
                
                # Lasso-like penalty
                L1penalty = self.LAMBDA * tf.reduce_sum(tf.abs(W))
                
                # Compute the L2 penalty (ridge-like)
                L2penalty = self.LAMBDA * tf.reduce_sum(W ** 2)
                    
                # Combine L1 and L2 penalty terms
                P = 0.5 * (self.ALPHA * L1penalty + (1 - self.ALPHA) * L2penalty)
            
            return P
        
        
        with tf.variable_scope("loss"):
    
            cumSum = tf.cast(tf.Variable([0.0]), tf.float32)
            Idx = tf.cast(tf.Variable(0), tf.int32)
            
            # Go through all uncensored cases and add to cumulative sum
            c = lambda Idx, cumSum: tf.less(Idx, tf.cast(tf.size(self.T)-1, tf.int32))
            b = lambda Idx, cumSum: _add_if_observed(Idx, cumSum)
            Idx, cumSum = tf.while_loop(c, b, [Idx, cumSum])
            
            # cost is negative weighted log likelihood
            self.cost = -cumSum
            
            # Add elastic-net penalty
            self.cost = self.cost + _penalty(self.W)

    #%%========================================================================
    #  Optimizer
    #==========================================================================

    def add_optimizer(self):
        
        """
        Adds optimizer to computational graph        
        """
        
        with tf.variable_scope("optimizer"):

            # Define optimizer and minimize loss
            if self.OPTIM == "RMSProp":
                self.optimizer = tf.train.RMSPropOptimizer(self.LEARN_RATE).\
                    minimize(self.cost)
                    
            elif self.OPTIM == "GD":
                self.optimizer = tf.train.GradientDescentOptimizer(self.LEARN_RATE).\
                    minimize(self.cost)
                    
            elif self.OPTIM == "Adam":
                self.optimizer = tf.train.AdamOptimizer(self.LEARN_RATE).\
                    minimize(self.cost)

        # Merge all summaries for tensorboard
        self.tbsummaries = tf.summary.merge_all()


#%%############################################################################ 
#%%############################################################################ 
#%%############################################################################
#%%############################################################################

#%%============================================================================
# test methods - quick and dirty
#==============================================================================

if __name__ == '__main__':
    
    #%%========================================================================
    # Prepare inputs
    #==========================================================================

    import os
    import sys
    
    def conditionalAppend(Dir):
        """ Append dir to sys path"""
        if Dir not in sys.path:
            sys.path.append(Dir)
    
    cwd = os.getcwd()
    conditionalAppend(cwd+"/../")
    import SurvivalUtils as sUtils
    
    import numpy as np
    from scipy.io import loadmat
    
    print("Loading and preprocessing data.")
    
    # Load data

    projectPath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
    #projectPath = "/home/mtageld/Desktop/KNN_Survival/"

    dpath = projectPath + "Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
    #dpath = projectPath + "Data/SingleCancerDatasets/GBMLGG/Brain_Gene.mat"
    #dpath = projectPath + "Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat"
    
    Data = loadmat(dpath)
    
    Features = np.float32(Data['Integ_X'])
    #Features = np.float32(Data['Gene_X'])
    
    N, D = Features.shape
    
    if np.min(Data['Survival']) < 0:
        Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1
    
    Survival = np.int32(Data['Survival']).reshape([N,])
    Censored = np.int32(Data['Censored']).reshape([N,])
    fnames = Data['Integ_Symbs']
    #fnames = Data['Gene_Symbs']
    
    RESULTPATH = projectPath + "Results/tmp/"
    MONITOR_STEP = 10
    description = "GBMLGG_Integ_"
    
    #  Preprocessing  
    #%%============================================================================
    
    # remove zero-variance features
    fvars = np.std(Features, 0)
    keep = fvars > 0
    Features = Features[:, keep]
    fnames = fnames[keep]
    
    ## Limit N (for prototyping) ----
    #n = 100
    #Features = Features[0:n, :]
    #Survival = Survival[0:n]
    #Censored = Censored[0:n]
    #--------------------------------    
    
    # *************************************************************
    # Z-scoring survival to prevent numerical errors
    Survival = (Survival - np.mean(Survival)) / np.std(Survival)
    # *************************************************************
    
    #  Separate out validation set   
    #%%============================================================================
    
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    
    N_valid = 100
    valid_idx = idxs[0:N_valid]
    train_idxs = idxs[N_valid:]
    
    Features_valid = Features[valid_idx, :]
    Survival_valid = Survival[valid_idx]
    Censored_valid = Censored[valid_idx]
    
    Features = Features[train_idxs, :]
    Survival = Survival[train_idxs]
    Censored = Censored[train_idxs]
    
    N, D = Features.shape # after feature removal
    
    #  Getting at-risk groups
    #%%============================================================================
    
    # Getting at-risk groups (trainign set)
    Features, Survival, Observed, at_risk = \
      sUtils.calc_at_risk(Features, Survival, 1-Censored)
      
    # Getting at-risk groups (validation set)
    Features_valid, Survival_valid, Observed_valid, at_risk_valid = \
      sUtils.calc_at_risk(Features_valid, Survival_valid, 1-Censored_valid)
      
    
    #%%============================================================================
    # Define relevant methods
    #==============================================================================
    
    import matplotlib.pylab as plt
    
    def _plotMonitor(arr, title, xlab, ylab, savename, arr2 = None):
                            
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
    
    #%%============================================================================
    #  Define computational graph   
    #==============================================================================

    graph_params = {'dim_input' : D,
                    'ALPHA': 0.1,
                    'LAMBDA': 1.0,
                    'OPTIM' : 'Adam',
                    'LEARN_RATE' : 0.01,
                    'PIJ_LOOP' : True,
                    }
    
    g = comput_graph(**graph_params)
    
    #%%============================================================================
    #   Run session
    #==============================================================================
    
    print("Running session.")
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        # for tensorboard visualization
        train_writer = tf.summary.FileWriter(RESULTPATH + '/tensorboard', sess.graph)
    
        feed_dict = {g.X_input: Features,
                     g.T: Survival,
                     g.O: Observed,
                     g.At_Risk: at_risk,
                     }
                   
        feed_dict_valid = {g.X_input: Features_valid,
                           g.T: Survival_valid,
                           g.O: Observed_valid,
                           g.At_Risk: at_risk_valid,
                           }
    
        costs = []
        costs_valid = []
        epochs = 0
        
        try: 
            while True:
                
                _, cost = sess.run([g.optimizer, g.cost], feed_dict = feed_dict)
                cost_valid = g.cost.eval(feed_dict = feed_dict_valid)

                # Normalize cost for sample size
                cost = cost / N
                cost_valid = cost_valid / N_valid
                
                print("epoch {}, cost_train = {}, cost_valid = {}"\
                        .format(epochs, cost, cost_valid))
                        
                # update costs
                costs.append([epochs, cost])
                costs_valid.append([epochs, cost_valid])
                
                # monitor
                if (epochs % MONITOR_STEP == 0) and (epochs > 0):
                    
                    cs = np.array(costs)
                    cs_valid = np.array(costs_valid)
                    
                    _plotMonitor(arr= cs, arr2= cs_valid[:,1],
                                 title= "cost vs. epoch", 
                                 xlab= "epoch", ylab= "cost", 
                                 savename= RESULTPATH + 
                                 description + "cost.png")
                
                epochs += 1
                
        except KeyboardInterrupt:
            
            print("\nFinished training model.")
            print("Obtaining final results.")
            
            W, B, X_transformed = sess.run([g.W, g.B, g.X_transformed], 
                                           feed_dict = feed_dict_valid)
                                           
            #X_transformed = g.X_transformed.eval(feed_dict = feed_dict)
            #W = np.ones(2)
                
            
            # Save model
            #self.save()
            
    #%%============================================================================
    # Rank features
    #==============================================================================

        
    def rankFeats(W, rank_type = "weights"):
        
        """ ranks features by feature weights or variance after transform"""
        
        print("Ranking features by " + rank_type)
    
        fidx = np.arange(D).reshape(D, 1)
        
        if rank_type == 'weights':
            # rank by feature weight
            ranking_metric = W.reshape(D, 1)
        elif rank_type == 'stdev':
            # rank by variance after transform
            ranking_metric = np.std(X_transformed, 0).reshape(D, 1)
        
        ranking_metric = np.concatenate((fidx, ranking_metric), 1)      
    
        # Plot feature weights/variance
        if D <= 500:
            n_plot = ranking_metric.shape[0]
        else:
            n_plot = 500
        _plotMonitor(ranking_metric[0:n_plot,:], 
                     "feature " + rank_type, 
                     "feature_index", rank_type, 
                     RESULTPATH + description + "feat_"+rank_type+"_.png")
        
        
        # rank features
        
        if rank_type == "weights":
            # sort by absolute weight but keep sign
            ranking = ranking_metric[np.abs(ranking_metric[:,1]).argsort()][::-1]
        elif rank_type == 'stdev':    
            ranking = ranking_metric[ranking_metric[:,1].argsort()][::-1]
        
        fnames_ranked = fnames[np.int32(ranking[:,0])].reshape(D, 1)
        fw = ranking[:,1].reshape(D, 1) 
        fnames_ranked = np.concatenate((fnames_ranked, fw), 1)
        
        # save results
        
        savename = RESULTPATH + description + rank_type + "_ranked.txt"
        with open(savename,'wb') as f:
            np.savetxt(f,fnames_ranked,fmt='%s', delimiter='\t')


    rankFeats(np.diag(W), rank_type = "weights")
    rankFeats(W, rank_type = "stdev")