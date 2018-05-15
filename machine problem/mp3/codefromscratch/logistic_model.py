"""logistic model class for binary classification."""

import numpy as np
import copy

class LogisticModel(object):

    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of W is the bias term,
            self.W = [Bias, W1, W2, W3, ...]
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
        self.W = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            self.W = np.zeros(ndims+1)
        elif W_init == 'ones':
            self.W = np.ones(ndims+1)
        elif W_init == 'uniform':
            self.W = np.random.uniform(low=0, high=1, size=ndims+1)
        elif W_init == 'gaussian':
            self.W = np.random.normal(loc=0, scale=0.1, size=ndims+1)
        else:
            print ('Unknown W_init ', W_init)

    def save_model(self, weight_file):
        """ Save well-trained weight into a binary file.
        Args:
            weight_file(str): binary file to save into.
        """
        self.W.astype('float32').tofile(weight_file)
        print ('model saved to', weight_file)

    def load_model(self, weight_file):
        """ Load pretrained weghit from a binary file.
        Args:
            weight_file(str): binary file to load from.
        """
        self.W = np.fromfile(weight_file, dtype=np.float32)
        print ('model loaded from', weight_file)


    def forward(self, X):
        """ Forward operation for logistic models.
            Performs the forward operation, and return probability score (sigmoid).
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): probability score of (label == +1) for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # dim = self.ndims+1
        scores = []
        w = self.W
        for i in range(len(X)):
            x = X[i]
            wTx = np.dot(w.T,x)
            score = 1/(1+np.exp(-wTx))
            scores.append(score)
        scores = np.array(scores)
        return scores

    def backward(self, Y_true, X):
        """ Backward operation for logistic models.
            Compute gradient according to the probability loss on lecture slides
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
        Returns:
            (numpy.ndarray): gradients of self.W
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        g_loss = np.zeros(self.ndims+1)
        w = self.W
        for i in range(len(X)):
            x = X[i]
            y = Y_true[i]
            g_loss += (-np.exp(-y*np.dot(w.T,x))*y*x)/(1+np.exp(-y*np.dot(w.T,x)))

        return g_loss

    def classify(self, X):
        """ Performs binary classification on input dataset.
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): predicted label = +1/-1 for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        y_hat = self.forward(X)
        w = self.W
        predicts = []
        for i in range(len(X)):
            x = X[i]
            predict = y_hat[i]
            if (predict >= 0.5):
                predicts.append(1)
            else:
                predicts.append(-1)
        return np.array(predicts)

    def fit(self, Y_true, X, learn_rate, max_iters):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            learn_rate: learning rate for gradient descent
            max_iters: maximal number of iterations
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        w1 = copy.deepcopy(self.W)
        for i in range(max_iters):
            g_loss = self.backward(Y_true, X)
            self.W -= learn_rate*g_loss
            if (np.linalg.norm(self.W-w1) == 0):
                break
            w1 = copy.deepcopy(self.W)
        # save_model("trained weights.np")
