import numpy as np
from RegularizerFunction import *


class Regularizer(RegularizerFunction):

    def forward(self):
        """
        Returns:
          Scalar, loss due to the L2 regularization.
        """
        # regularization term loss portion of L1 loss = lambda * sum(abs(weight))
        return self.weight_decay * np.sum(np.abs(self.weights))

    def backward_params(self):
        """
        Returns:
          Gradient of the L1 loss with respect to the regularized weights.
        """
        grad_weights = self.weight_decay * np.sign(self.weights)
        return grad_weights
