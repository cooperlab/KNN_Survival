import numpy as np
from neuralnetwork import NeuralNetwork
from perceptron import Perceptron
from random import shuffle
import scipy.io as sio
import os


if __name__ == "__main__":

    # load path
    data_path = '/Users/rafihaque/machinelearningproject/neuralnetwork/'
    data = sio.loadmat(data_path+'gbm_org.mat')
    save = 'gbm_org_test.mat'
    data_struct = data['newdata']

    x = data_struct['features'][0][0]
    y = data_struct['survival'][0][0]
    tr = data_struct['tr'][0][0]
    te = data_struct['te'][0][0]
    censored = data_struct['censored'][0][0]
    subs = data_struct['subs'][0][0]

    # paramters
    num_folds  = 5
    num_nodes = 5
    num_feats  = x.shape[1]

    # outputs
    y_act = []
    y_hat_p = []
    y_hat_nn = []
    y_act_nn = []
    group = []
    cens = []


    for fold in range(num_folds):
        print "FOLD:", fold

        # store ytrain and ytest
        labels = np.argmax(y, axis=1)
        tr_obs = tr[fold].astype(bool)
        te_obs = te[fold].astype(bool)

        # store actual labels
        y_act.append(labels[te_obs])

        # store censored
        cens.append(censored[te_obs])
        group.append(subs[te_obs])

        # train and test perceptron
        perc = Perceptron(num_feats=num_feats,learn_rate=0.1)
        perc.train(x[tr_obs],y[tr_obs])
        print np.argmax(perc.predict(x[te_obs]),axis=1)
        y_hat_p.append(np.argmax(perc.predict(x[te_obs]), axis=1))
        perc.close()




    test = {}
    test['te'] = te
    test['y_act'] = y_act
    test['y_act_nn'] = y_act
    test['y_hat_p'] = y_hat_p
    test['y_hat_nn'] = y_hat_nn
    test['cens'] = cens
    test['group'] = group



    sio.savemat(data_path + save,test)
