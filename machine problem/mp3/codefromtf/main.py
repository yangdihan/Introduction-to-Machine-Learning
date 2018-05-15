"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.001
max_iters = 50

def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset','indexing.txt')
    A, T = read_dataset_tf('../data/trainset','indexing.txt')
    # Initialize model.
    ndims = 16
    model = LogisticModel_TF(ndims, W_init='gaussian')
    # Build TensorFlow training graph
    model.build_graph(learn_rate)
    # Train model via gradient descent.
    predicts = model.fit(T, A, max_iters)
    # Compute classification accuracy based on the return of the "fit" method
    correct = 0
    total = len(predicts)
    for i in range(total):
        if (predicts[i] == T[i]):
            correct += 1
    acc = correct/total
    print(acc)



if __name__ == '__main__':
    tf.app.run()
