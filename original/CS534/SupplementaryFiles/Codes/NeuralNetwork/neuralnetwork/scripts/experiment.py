
from neuralnetwork import NeuralNetwork
from random import shuffle
import scipy.io as sio



if __name__ == "__main__":

    # load transformed features
    data_path = '/Users/rafihaque/machinelearningproject/neuralnetwork/'
    data = sio.loadmat(data_path+'gbm_mod.mat')
    save = 'gbm_mod_nn.mat'
    data_struct = data['newdata']
    x = data_struct['features'][0][0]
    y = data_struct['survival2'][0][0]


    # paramters
    num_nodes = 5
    cv = 0.8
    num_folds  = 5
    num_obs = x.shape[0]
    num_feats  = x.shape[1]
    num_tr_obs = int(round(cv * num_obs))
    num_te_obs = num_obs-num_tr_obs-1

    # outputs
    y_train = []
    y_test = []
    yn_test = []

    for fold in range(num_folds):
        print "FOLD:", fold
        # randomize observation indices
        all_obs = range(num_obs)
        shuffle(all_obs)
        train_obs = all_obs[:num_tr_obs]
        test_obs = all_obs[num_tr_obs + 1:]

        # store ytrain and ytest
        y_train.append(y[train_obs])
        y_test.append(y[test_obs])

        # train and test neural network
        for node in range(num_nodes):
            print " NODE:",3*node + 1
            nn = NeuralNetwork(num_feats=num_feats, num_nodes=3*node + 1, learn_rate=0.1)
            nn.train(x[train_obs],y[train_obs])
            yn_test.append(nn.predict(x[test_obs]))
            nn.close()

    test = {}
    test['y_train'] = y_train
    test['y_test'] = y_test
    test['yn_test'] = yn_test
    sio.savemat(data_path + save,test)
