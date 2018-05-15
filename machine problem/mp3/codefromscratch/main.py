"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.01
max_iters = 200

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')
    A, T = read_dataset('../data/trainset','indexing.txt')
    # Initialize model.
    ndims = 16
    model = LogisticModel(ndims, W_init='gaussian')
    # Train model via gradient descent.
    model.fit(T, A, learn_rate, max_iters)
    # Save trained model to 'trained_weights.np'
    model.save_model('trained_weights.np')
    # Load trained model from 'trained_weights.np'
    model.load_model('trained_weights.np')
    # Try all other methods: forward, backward, classify, compute accuracy
    predicts = model.classify(A)
    correct = 0
    total = len(predicts)
    for i in range(total):
        if (predicts[i] == T[i]):
            correct += 1
    acc = correct/total
    print(acc)
