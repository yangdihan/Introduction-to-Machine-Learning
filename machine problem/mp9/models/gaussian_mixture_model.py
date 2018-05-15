"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from copy import deepcopy

class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=0.1):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar
        # Randomly Initialize model parameters
        self._mu = np.random.rand(n_components, n_dims)  # np.array of size (n_components, n_dims)
        # Initialized with uniform distribution.
        self._pi = np.random.uniform(low=0.0, high=1.0, size=(n_components, 1))  # np.array of size (n_components, 1)
        # Initialized with identity.
        self._sigma = np.zeros((n_components, n_dims, n_dims))  # np.array of size (n_components, n_dims, n_dims)
        for i in range(n_components):
            self._sigma[i] = 10000*np.identity(n_dims)
        return

    def k_mean_initialize(self, x):
        N,dim = x.shape
        K = self._n_components
        C = np.random.uniform(low=np.min(x), high=np.max(x), size=(K,dim))
        for i in range(50):
            D = []
            for k in range(K):
                D.append([])

            for data in x:
                points = deepcopy(data).astype(float)
                cluster_index = np.argmin(np.linalg.norm(C-points,axis=1))
                D[cluster_index].append(points)

            for k in range(K):
                if (len(D[k]) == 0):
                    continue
                else:
                    C[k] = np.average(np.array(D[k]),axis=0)
        self._mu=C
        return

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        self.k_mean_initialize(x)
        for i in range(self._max_iter):
            print(i)
            self._m_step(x, self._e_step(x))

        return

    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        return self.get_posterior(x)


    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        N = len(x)

        self._pi = np.average(z_ik,axis=0).reshape(self._n_components,1)
        self._mu = np.divide(np.dot(z_ik.T,x)/N,self._pi)

        sigma = []
        for k in range(self._n_components):
            sum = np.zeros((self._n_dims,self._n_dims))
            for i in range(N):
                diff = (x[i]-self._mu[k]).reshape(self._n_dims,1)
                sum += z_ik[i,k]*np.matmul(diff,diff.T)
            sigma.append(sum/N/self._pi[k] + self._reg_covar*np.identity(self._n_dims))
        self._sigma = np.array(sigma)

        return


    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N,, n_components).
        """
        ret = []
        for k in range(self._n_components):
            mu_k = self._mu[k] #dim
            sigma_k = self._sigma[k] # dim*dim
            p = self._multivariate_gaussian(x, mu_k, sigma_k)
            # if (np.linalg.norm(p) < 1e-50):
            #     print(np.max(p))
            #     p += 1e-6
            ret.append(self._pi[k] * p)
            # ret.append(self._pi[k] * self._multivariate_gaussian(x, mu_k, sigma_k))

        return np.array(ret).T

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        return np.sum(self.get_conditional(x),axis=1)

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        cond = self.get_conditional(x)
        marg = self.get_marginals(x).reshape(len(x),1)
        return np.divide(cond,marg)


    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)


    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """

        self.cluster_label_map = []
        # self.fit(x)
        zik = self.get_posterior(x)
        Labels = np.unique(y)
        pool = np.zeros((self._n_components,len(Labels)))
        for i in range(len(y)):
            cluster = np.argmax(zik[i])
            label = int(y[i])
            pool[cluster,list(Labels).index(label)] += 1
        for cluster_vote in pool:
            common_label = Labels[np.argmax(cluster_vote)]
            self.cluster_label_map.append(common_label)

        return

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """

        z_ik = self.get_posterior(x)
        y_hat = []
        for p in z_ik:
            y_hat.append(self.cluster_label_map[np.argmax(p)])
        return np.array(y_hat)
