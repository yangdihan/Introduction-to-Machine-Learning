"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers
from sklearn.utils import shuffle as sf
import copy

def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    dataset = copy.deepcopy(data)
    X = dataset['image']
    Y = dataset['label']

    for i in range(num_steps):
        print(i)
        if shuffle:
            dataset['image'], dataset['label'] = sf(dataset['image'], dataset['label'])

        count = 0
        while count < len(Y):
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
    # Implementation here. (This function will not be graded.)
    f_predict = model.forward(x_batch)
    g = model.backward(f_predict, y_batch)
    model.w -= learning_rate*g
    return


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    # Set model.w
    # model.w = z
    model.w = z[:data['image'].shape[1]+1]
    return


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    # original
    # X = data['image']
    # Y = data['label']
    # dim = len(model.w)
    # N = len(Y)
    # P = model.w_decay_factor*np.identity(dim)
    # q = np.zeros((dim,1))
    # q[-1] = -0.5*model.w[-1]
    # ones = np.ones((N,1))
    # X = np.append(X, ones, 1)
    # G = -np.multiply(Y.reshape((N,1)),X)
    # h = -1*np.ones((N,1))

# change
    X = data['image']
    Y = data['label']
    dim = len(model.w)
    N = len(Y)
    P = model.w_decay_factor*np.identity(dim+N)
    q = np.zeros((dim+N,1))
    q[dim:-1] = 1
    ones = np.ones((N,1))
    X = np.append(X, ones, 1)
    G = -np.multiply(Y.reshape((N,1)),X)
    G = np.append(G,-np.identity(N),1)
    G_= np.append(np.zeros((N,dim)),-np.identity(N),1)
    G = np.append(G,G_,0)
    h = -1*np.ones((N,1))
    h = np.append(h,np.zeros((N,1)),0)

    # Implementation here.
    # P = None
    # q = None
    # G = None
    # h = None
    # y = data['label']
    # x = data['image']
    # newList = np.ones((x.shape[0], 1))
    # x = np.append(x, newList, axis=1)
    # N = x.shape[0]
    # dim = model.ndims+1
    # P = np.zeros((dim + N, dim + N))
    # for i in range(dim):
    #     P[i][i] = model.w_decay_factor
    #
    # q = np.zeros((dim + N, 1))
    # for i in range(dim, dim + N):
    #     q[i] = 1
    # G1 = -np.multiply(y.reshape((N,1)),x)
    # G2 = -np.eye(N)
    # G3 = np.zeros((N,dim))
    # G4 = -np.eye(N)
    # G12 = np.append(G1, G2, 1)
    # G34 = np.append(G3, G4, 1)
    # G = np.append(G12, G34, 0)
    # h1 = -np.ones((N,1))
    # h2 = np.zeros((N,1))
    # h = np.append(h1, h2, 0)
    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    X = data['image']
    Y = data['label']
    y_infer = model.forward(X)
    predicts = model.predict(y_infer)
    loss = model.total_loss(y_infer,Y)
    acc = 0
    for i in range(len(predicts)):
        if (predicts[i] == Y[i]):
            acc += 1
    acc /= len(y_infer)
    return loss, acc
