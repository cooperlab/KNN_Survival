import tensorflow as tf

class NeuralNetwork(object):
    def __init__(self, num_feats, num_nodes,learn_rate=0.5):

        # create session
        self.sess = tf.Session()

        # create placeholders for inputs
        self.x = self.placeholder('x',tf.float32,[None, num_feats])
        self.y = self.placeholder('y',tf.float32,[None, 2])

        # weight and bias variables for neural network
        num_nodes = 2
        self.w1 = tf.Variable(tf.zeros([num_feats, num_nodes]))
        self.w2 = tf.Variable(tf.zeros([num_nodes, num_nodes]))
        self.w3 = tf.Variable(tf.zeros([num_nodes, 2]))

        # create model
        self.yhat = self.model(self.x)

         # loss
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.yhat), reduction_indices=[1]))
        self.update = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(self.cross_entropy)

        # initialize all variables
        self.sess.run([tf.initialize_all_variables()])
        self.saver = tf.train.Saver()

    def model(self,x):
        self.layer1 = tf.matmul(self.x, self.w1)
        self.layer2 = tf.matmul(self.layer1,self.w3)
        return tf.nn.softmax(self.layer2)


    def get_value(self,x):
        return self.sess.run(self.layer1, feed_dict={
            self.x: x
        })


    def train(self,x,y):
        self.sess.run([self.update], feed_dict={
            self.x: x,
            self.y: y
        })

    def predict(self,testx):
        return self.sess.run(self.yhat, feed_dict={
            self.x: testx
        })

    def savemodel(self,path):
        self.saver.save(self.sess,path)

    def close(self):
        self.sess.close()

    def placeholder(self,name,type,shape):
        return tf.placeholder(name=name,dtype=type,shape=shape)






if __name__ == "__main__":
    pass