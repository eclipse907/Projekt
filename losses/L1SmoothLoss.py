import numpy as np
from losses.LossFunction import *


class L1SmoothLoss(LossFunction):

    def __init__(self, weights, weight_decay, name):
        """
        Args:
          weights: parameters which will be regularized
          weight_decay: lambda, regularization strength
          name: layer name
        """
        # this is still a reference to original tensor so don't change self.weights
        self.weights = weights
        self.weight_decay = weight_decay
        self.name = name

    def forward(self):
        """
         Returns:
          Scalar, smooth l1 loss.
        """
        # smooth L1 loss
        # (for abs(x) < 1/lambda) = 0.5 * lambda * sum(weight**2)
        # (for abs(x) >= 1/lambda) = abs(weight)-0.5/lambda
        lossAllWeights = np.where(np.abs(self.weights) < 1 / self.weight_decay,
                                  self.weight_decay * 0.5 * np.square(self.weights),
                                  np.abs(self.weights) - 0.5 / self.weight_decay)

        print(lossAllWeights)
        return np.sum(lossAllWeights)

    def backward_params(self):
        """
        Returns:
          Gradient of the smooth L1 loss with respect to the weights.
        """

        gradAllWeights = np.where(np.abs(self.weights) < 1 / self.weight_decay, self.weight_decay * self.weights,
                                  np.sign(self.weights))

        print(gradAllWeights)
        return [[self.weights, gradAllWeights], self.name]
