import numpy as np
from regularizers.Regularizer import *


class L2Regularizer(Regularizer):

    def forward(self):
        """
        Returns:
          Scalar, loss due to the L2 regularization.
        """
        # regularization term loss portion of L2 loss = lambda/2 * sum(weight**2)
        return self.weight_decay / 2 * np.sum(np.square(self.weights))

    def backward_params(self):
        """
        Returns:
          Gradient of the L2 loss with respect to the regularized weights.
        """
        grad_weights = self.weight_decay * self.weights
        #return [[self.weights, grad_weights], self.name]
        return grad_weights
