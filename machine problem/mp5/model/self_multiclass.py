import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            print(i)
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)


    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        for i in range(10):
            out = np.zeros(y.shape)
            for j in range(len(out)):
                if (y[j] == i):
                    out[j] = 1
            model = svm.LinearSVC(random_state = 12345).fit(X,out)
            binary_svm[i] = model
        return binary_svm

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        for a in range(10):
            for b in range(a+1,10):
                y_out = []
                x_out = []
                for i in range(len(y)):
                    if (y[i] == a):
                        y_out.append(0)
                        x_out.append(X[i])
                    if (y[i] == b):
                        y_out.append(1)
                        x_out.append(X[i])
                model = svm.LinearSVC(random_state = 12345).fit(np.array(x_out), np.array(y_out))
                binary_svm[(a,b)] = model
        return binary_svm

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = np.zeros((len(X),10))
        for clf in self.binary_svm:
            scores[:,clf] = self.binary_svm[clf].decision_function(X)
        return scores

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = np.zeros((len(X),10))
        for clf in self.binary_svm:
            a = clf[0]
            b = clf[1]
            votes = self.binary_svm[clf].predict(X)
            for i in range(len(votes)):
                if (votes[i] == 0):
                    scores[i,a] += 1
                if (votes[i] == 1):
                    scores[i,b] += 1
        return scores

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        K = len(W)
        N = len(y)
        loss1 = 0
        loss2 = 0
        for j in range(K):
            loss1 += np.sum(W[j]**2)
        loss1 *= 0.5
        for i in range(N):
            v = np.dot(X[i],W.T)
            v += 1
            v[y[i]] -= 1
            loss2 += np.max(v) - np.dot(W[y[i]],X[i])
        loss2 *= C
        return loss1+loss2

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        K = len(W)
        N = len(y)
        g = np.zeros(W.shape)
        for i in range(N):
            # for a single data point
            v = np.dot(X[i],W.T)
            v += 1
            v[y[i]] -= 1
            g[np.argmax(v)] += X[i]
            g[y[i]] -= X[i]
        g *= C
        g += W
        return g
