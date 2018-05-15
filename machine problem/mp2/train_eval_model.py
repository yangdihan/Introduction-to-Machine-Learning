"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression
from sklearn.utils import shuffle as sf

def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    dataset = processed_dataset
    X = dataset[0]
    Y = dataset[1]

    for i in range(num_steps):
        if shuffle:
            dataset[0], dataset[1] = sf(dataset[0],dataset[1])

        count = 0

        while count < len(dataset[0]):
            update_step(X[count : count+batch_size], Y[count : count+batch_size], model, learning_rate)
            count += batch_size

        update_step(X[count-batch_size : ], Y[count-batch_size : ], model, learning_rate)

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    f_predict = model.forward(x_batch)
    g = model.backward(f_predict, y_batch)
    model.w -= learning_rate*g

    return


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    X = processed_dataset[0]
    one = np.ones((X.shape[0],1))
    X = np.append(X,one,1)
    Y = processed_dataset[1]
    plus = model.w_decay_factor*np.identity(len(np.dot(X.T,X)))
    a = np.linalg.inv(np.dot(X.T,X)+plus)
    b = np.dot(a,X.T)
    c = np.dot(b,Y)
    model.w = c

    return

def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    X = processed_dataset[0]
    Y = processed_dataset[1]
    y_infer = model.forward(X)
    loss = model.total_loss(y_infer,Y)
    acc = 1 - np.mean(abs(Y - y_infer)/Y)
    return loss
