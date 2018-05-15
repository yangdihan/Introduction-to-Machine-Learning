"""logistic model class for binary classification."""
import tensorflow as tf
import numpy as np

class LogisticModel_TF(object):

    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of Weight is the bias term,
            Weight = [Bias, W1, W2, W3, ...]
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W0 = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            self.W0 = tf.zeros((ndims+1,1))
        elif W_init == 'ones':
            self.W0 = tf.ones((ndims+1,1))
        elif W_init == 'uniform':
            self.W0 = tf.random_uniform((ndims+1,1))
        elif W_init == 'gaussian':
            self.W0 = tf.random_normal((ndims+1,1),mean=0.0, stddev=0.1)
        else:
            print ('Unknown W_init ', W_init)


    def build_graph(self, learn_rate):
        """ build tensorflow training graph for logistic model.
        Args:
            learn_rate: learn rate for gradient descent
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        self.x = tf.placeholder(tf.float32)
        self.y_true = tf.placeholder(tf.float32)
        self.W = tf.Variable(self.W0)
        wTx = tf.matmul(self.x,self.W)
        # forward score
        g = tf.sigmoid(wTx)
        squared_deltas = tf.square(tf.subtract(self.y_true,g))
        loss = tf.reduce_sum(squared_deltas)
        # optimizer definition
        optimizer = tf.train.GradientDescentOptimizer(learn_rate)
        self.train = optimizer.minimize(loss)


    def fit(self, Y_true, X, max_iters):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,1)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            max_iters: maximal number of training iterations
            ......: append as many arguments as you want
        Returns:
            (numpy.ndarray): sigmoid output from well trained logistic model, used for classification
                             with a dimension of (# of samples, 1)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        init = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(init)
            # train
            for step in range(max_iters):
                print(step)
                session.run(self.train, feed_dict={self.x:X, self.y_true:Y_true})
            # use optimized weight to make predictions
            w_opt = session.run(self.W)
            X = tf.cast(X,tf.float32)
            predicts = session.run(tf.sigmoid(tf.matmul(X,w_opt)))

        # make the return
        output = []
        for score in predicts:
            if (score >= 0.5):
                output.append(1)
            else:
                output.append(0)

        return np.array(output)
