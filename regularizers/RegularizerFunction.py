from abc import ABC, abstractmethod


class RegularizerFunction(ABC):
    def __init__(self, model, paramsModule):
        """
        Args:
          weights: parameters which will be regularized
          weight_decay: lambda, regularization strength
          name: layer name
        """
        # this is still a reference to original tensor so don't change self.weights
        self.weights = [model.W1, model.W2]
        self.weight_decay = paramsModule.weight_decay

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
