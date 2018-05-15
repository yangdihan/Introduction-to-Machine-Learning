"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation based on the loss in total_loss.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1, 1).
        """
        # Implementation here.
        reg_grad = self.w_decay_factor * self.w
        y_grad = np.zeros(len(y))
        for i in range(len(y_grad)):
            if (1-y[i]*f[i] > 0):
                y_grad[i] = -y[i]
        loss_grad = np.matmul(y_grad,self.x)
        total_grad = reg_grad + loss_grad
        return total_grad

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor/2*||w||^2

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """
        # hinge_loss = None
        # l2_loss = None
        # Implementation here.
        l2_loss = 0.5*self.w_decay_factor*(np.linalg.norm(self.w))**2
        hinge_loss = 0
        for i in range(len(f)):
            hinge_loss += max(0,1-y[i]*f[i])
        total_loss = hinge_loss + l2_loss
        return total_loss

    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,1). Tie break 0 to 1.0.
        """
        # Implementation here.
        y_predict = []
        for score in f:
            if (score >= 0):
                y_predict.append(1)
            else:
                y_predict.append(-1)

        return np.array(y_predict)
