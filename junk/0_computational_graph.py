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
                 transform_type = "linear",
                 ALPHA = 0.5,
                 LAMBDA = 1,
                 nn_params = {'DEPTH': 2},
                 OPTIM = 'GD',
                 LEARN_RATE = 0.01):
        
        """
        Instantiate a computational graph for survival NCA.
        
        Args:
        ------
        dim_input - no of features
        
        """
        
        print("Building computational graph for survival NCA.")        
        
        # set up instace attributes
        self.dim_input = dim_input
        self.transform_type = transform_type
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.nn_params = nn_params
        self.OPTIM = OPTIM
        self.LEARN_RATE = LEARN_RATE
        
        # clear lurking tensors
        tf.reset_default_graph()
        
        print("Adding placeholders.")
        self.add_placeholders()
        
        # fature space transform
        if transform_type == "linear":
            print("Adding linear feature transform.")
            self.add_linear_transform()
        elif transform_type == "ffNetwork":
            print("Adding ffNetwork transform.")
            self.add_ffNetwork(**nn_params)
            
        print("Adding regularized weighted log likelihood.")
        self.add_cost()
        
        print("Adding optimizer.")
        self.add_optimizer()
        
        

    #%%========================================================================
    # Random useful methods
    #==========================================================================
    
    def _variable_summaries(self, var):
        
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        #with tf.name_scope('stddev'):
        #  stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        #tf.summary.scalar('stddev', stddev)
        #tf.summary.scalar('max', tf.reduce_max(var))
        #tf.summary.scalar('min', tf.reduce_min(var))
        #tf.summary.histogram('histogram', var)


    #%%========================================================================
    # Add placeholders to graph  
    #==========================================================================
    
    def add_placeholders(self):
    
        """ Adds graph inputs as placeholders in graph """
        
        with tf.variable_scope("Inputs"):
        
            self.X_input = tf.placeholder("float", [None, self.dim_input], name='X_input')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') #for dropout
            
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
            self.w = tf.get_variable("weights", shape=[self.dim_input], 
                            initializer= tf.contrib.layers.xavier_initializer())
            
            self.b = tf.get_variable("biases", shape=[self.dim_input], 
                            initializer= tf.contrib.layers.xavier_initializer())
            
            # diagonalize and matmul
            W = tf.diag(self.w)
            self.X_transformed = tf.add(tf.matmul(self.X_input, W), self.b) 
    

    #%%========================================================================
    # Nonlinear feature transformation (feed forward network)
    #==========================================================================

    def add_ffNetwork(self, DEPTH = 2, MAXWIDTH = 200, 
                      NONLIN = "ReLU", LINEAR_READOUT = False,
                      DIM_OUT = None):
        """ 
        Adds a feedforward network to the computational graph,
        which performs a series of non-linear transformations to
        the input matrix and outputs a matrix of the same dimss.
        """
        
        # Define sizes of weights and biases
        #======================================================================
        
        if DIM_OUT is None:
            # do not do any dimensionality reduction
            DIM_OUT = self.dim_input
        
        dim_in = self.dim_input
        
        if DEPTH == 1:
            dim_out = DIM_OUT
        else:
            dim_out = MAXWIDTH
        
        weights_sizes = {'layer_1': [dim_in, dim_out]}
        biases_sizes = {'layer_1': [dim_out]}
        dim_in = dim_out
        
        # intermediate layers
        if DEPTH > 2:
            for i in range(2, DEPTH):                
                dim_out = int(dim_out)
                weights_sizes['layer_{}'.format(i)] = [dim_in, dim_out]
                biases_sizes['layer_{}'.format(i)] = [dim_out]
                dim_in = dim_out
        
        # last layer
        if DEPTH > 1:
            weights_sizes['layer_{}'.format(DEPTH)] = [dim_in, DIM_OUT]
            biases_sizes['layer_{}'.format(DEPTH)] = [DIM_OUT]
        
        # Define a layer
        #======================================================================
        
        def _add_layer(layer_name, Input, APPLY_NONLIN = True,
                       Mode = "Encoder", Drop = True):
            
            """ adds a single fully-connected layer"""
            
            with tf.variable_scope(layer_name):
                #
                # initialize using xavier method
                #
                m_w = weights_sizes[layer_name][0]
                n_w = weights_sizes[layer_name][1]
                m_b = biases_sizes[layer_name][0]
                
                w = tf.get_variable("weights", shape=[m_w, n_w], 
                                    initializer= tf.contrib.layers.xavier_initializer())
                self._variable_summaries(w)
             
                b = tf.get_variable("biases", shape=[m_b], 
                                    initializer= tf.contrib.layers.xavier_initializer())
                
                #
                # Do the matmul and apply nonlin
                # 
                if Mode == "Encoder":
                    l = tf.add(tf.matmul(Input, w),b) 
                elif Mode == "Decoder":
                    l = tf.matmul(tf.add(Input,b), w) 
                
                if APPLY_NONLIN:
                    if NONLIN == "Sigmoid":  
                        l = tf.nn.sigmoid(l, name= 'activation')
                    elif NONLIN == "ReLU":  
                        l = tf.nn.relu(l, name= 'activation')
                    elif NONLIN == "Tanh":  
                        l = tf.nn.tanh(l, name= 'activation') 
                    #tf.summary.histogram('activations', l)
                
                # Dropout
                if Drop:
                    l = tf.nn.dropout(l, self.keep_prob)
                    
                return l
        
        # Add the layers
        #======================================================================
            
        with tf.variable_scope("ffNetwork"):
            
            l_in = self.X_input
            
            for i in range(1, DEPTH):
                 l_in = _add_layer("layer_{}".format(i), l_in)
                 
            # outer layer (prediction)
            self.X_transformed = _add_layer("layer_{}".format(DEPTH), l_in,
                                APPLY_NONLIN = not(LINEAR_READOUT),
                                Drop=False)


    #%%========================================================================
    # Get Pij 
    #==========================================================================

    def _get_Pij(self):
        
        """ 
        Calculate Pij, the probability that j will be chosen 
        as i's neighbor, for all i's
        
        Inspired by: https://github.com/RolT/NCA-python
        """
        
        with tf.name_scope("getting_Pij"):
            
            # transpose so that feats are in rows
            AX = tf.transpose(self.X_transformed)
            
            # Expand dims of AX to [n_features, n_samples, n_samples], where
            # each "channel" in the third dimension is the difference between
            # one sample and all other samples
            normAX = AX[:, :, None] - AX[:, None, :]
            
            # Now get the euclidian distance between
            # every patient and all others -> [n_samples, n_samples]
            #normAX = tf.norm(normAX, axis=0)
            normAX = tf.reduce_sum(normAX ** 2, axis=0)
            
            # Calculate Pij, the probability that j will be chosen 
            # as i's neighbor, for all i's. Pij has shape
            # [n_samples, n_samples] and ** is NOT symmetrical **.
            # Because the data is normalized using softmax, values
            # add to 1 in rows, that is i (central patients) are
            # represented in rows
            denomSum = tf.reduce_sum(tf.exp(-normAX), axis=0)
            epsilon = 1e-5
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
                L1penalty = self.LAMBDA * tf.reduce_sum(tf.abs(W), axis=0)
                
                # Compute the L2 penalty (ridge-like)
                L2penalty = self.LAMBDA * tf.reduce_sum(W ** 2, axis=0)
                    
                # Combine L1 and L2 penalty terms
                P = 0.5 * (self.ALPHA * L1penalty + (1 - self.ALPHA) * L2penalty)
            
            return P
        
        
        with tf.variable_scope("loss"):
    
            cumSum = tf.cast(tf.Variable([0.0]), tf.float32)
            Idx = tf.cast(tf.Variable(0), tf.int32)
            
            # Doing the following admittedly odd step because tensorflow's loop
            # requires both the condition and body to have same number of inputs
            def _cmp_pred(Idx, cumSum):
                return tf.less(Idx, tf.cast(tf.size(self.T)-1, tf.int32))
            
            # Go through all uncensored cases and add to cumulative sum
            c = lambda Idx, cumSum: _cmp_pred(Idx, cumSum)
            b = lambda Idx, cumSum: _add_if_observed(Idx, cumSum)
            Idx, cumSum = tf.while_loop(c, b, [Idx, cumSum])
            
            # cost is negative weighted log likelihood
            self.cost = -cumSum
            
            # Add elastic-net penalty
            if self.transform_type == "linear":
                self.cost = self.cost + _penalty(self.w)


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
    
    KEEP_PROB = 0.9
    
    # Load data
    #dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Integ.mat"
    dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/GBMLGG/Brain_Gene.mat"
    #dpath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Data/SingleCancerDatasets/BRCA/BRCA_Integ.mat"
    
    Data = loadmat(dpath)
    
    #Features = np.float32(Data['Integ_X'])
    Features = np.float32(Data['Gene_X'])
    
    N, D = Features.shape
    
    if np.min(Data['Survival']) < 0:
        Data['Survival'] = Data['Survival'] - np.min(Data['Survival']) + 1
    
    Survival = np.int32(Data['Survival']).reshape([N,])
    Censored = np.int32(Data['Censored']).reshape([N,])
    #fnames = Data['Integ_Symbs']
    fnames = Data['Gene_Symbs']
    
    # remove zero-variance features
    fvars = np.std(Features, 0)
    keep = fvars > 0
    Features = Features[:, keep]
    fnames = fnames[keep]
    N, D = Features.shape # after feature removal
    
    # Getting at-risk groups (trainign set)
    Features, Survival, Observed, at_risk = \
      sUtils.calc_at_risk(Features, Survival, 1-Censored)
      
    ## Limit N (for prototyping)  
    #n = 100
    #Features = Features[0:n, :]
    #Survival = Survival[0:n]
    #Observed = Observed[0:n]
    #at_risk = at_risk[0:n]
    
    # *************************************************************
    # Z-scoring survival to prevent numerical errors
    Survival = (Survival - np.mean(Survival)) / np.std(Survival)
    # *************************************************************
    
    #%%============================================================================
    # Define relevant methods
    #==============================================================================
    
    import matplotlib.pylab as plt
    
    def _plotMonitor(arr, title, xlab, ylab, savename):
                            
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
    
    RESULTPATH = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/Results/tmp/"
    MONITOR_STEP = 10
    description = "GBMLGG_Gene_"
    
    #%%============================================================================
    #  Define computational graph   
    #==============================================================================

    nn_params = {'DEPTH' : 2, 
                 'MAXWIDTH' : 200, 
                 'NONLIN' : "Tanh", 
                 'LINEAR_READOUT' : False,
                 'DIM_OUT' : None,
                 }
    
    graph_params = {'dim_input' : D,
                    'transform_type' : "linear",
                    'ALPHA': 0.1,
                    'LAMBDA': 1.0,
                    'nn_params' : nn_params,
                    'OPTIM' : 'Adam',
                    'LEARN_RATE' : 0.01,
                    }
    
    g = comput_graph(**graph_params)
    
    #%%============================================================================
    #   Run session
    #==============================================================================
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        # for tensorboard visualization
        train_writer = tf.summary.FileWriter(RESULTPATH + '/tensorboard', sess.graph)
    
        feed_dict={g.X_input: Features,
                   g.T: Survival,
                   g.O: Observed,
                   g.At_Risk: at_risk,
                   g.keep_prob: KEEP_PROB}
    
        costs = []
        epochs = 0
        
        try: 
            while True:
                
                _, cost = sess.run([g.optimizer, g.cost], feed_dict = feed_dict)
                
                print("epoch {}, cost = {}".format(epochs, cost))
        
                # update costs
                costs.append([epochs, cost])
                
                # monitor
                if (epochs % MONITOR_STEP == 0) and (epochs > 0):
                    cs = np.array(costs)
                    _plotMonitor(arr= cs, title= "cost vs. epoch", 
                                 xlab= "epoch", ylab= "objective", 
                                 savename= RESULTPATH + 
                                 description + "cost.svg")
                
                epochs += 1
                
        except KeyboardInterrupt:
            
            print("\nFinished training model.")
            print("Obtaining final results.")
            
            if graph_params['transform_type'] == 'linear':
                W, B, X_transformed = sess.run([g.w, g.b, g.X_transformed], 
                                               feed_dict = feed_dict)
            
            # Save model
            #self.save()
            
    #%%============================================================================
    # Rank features
    #==============================================================================
    
    fidx = np.arange(D).reshape(D, 1)
    W = W.reshape(D, 1)
    fweights = np.concatenate((fidx, W), 1)   
    
    # Plot feature weights
    _plotMonitor(fweights, "feature_weights", "feature_index", "weight", 
                 RESULTPATH + description + "featweights.svg")
    
    # rank by absolute feature weight
    fweights = fweights[np.abs(fweights[:,1]).argsort()][::-1]
    fnames_ranked = fnames[np.int32(fweights[:,0])].reshape(D, 1)
    fw = fweights[:,1].reshape(D, 1) 
    fnames_ranked = np.concatenate((fnames_ranked, fw), 1)
    
    # save results
    
    savename = RESULTPATH + description + "featnames_ranked.txt"
    with open(savename,'wb') as f:
        np.savetxt(f,fnames_ranked,fmt='%s', delimiter='\t')