from abc import ABC, abstractmethod


class RegularizerFunction(ABC):
    def __init__(self, weights, weight_decay):
        """
        Args:
          weights: parameters which will be regularized
          weight_decay: lambda, regularization strength
          name: layer name
        """
        # this is still a reference to original tensor so don't change self.weights
        self.weights = weights
        self.weight_decay = weight_decay

    @abstractmethod
    def forward(self):
        """
         Returns:
          Scalar, loss due to the regularization.
        """
        pass

    @abstractmethod
    def backward_params(self):
        """
        Returns:
          Gradient of the L2 loss with respect to the regularized weights.
        """
        pass
