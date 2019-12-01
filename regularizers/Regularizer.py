class Regularizer():
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
          Scalar, loss due to the regularization.
        """
        raise NotImplementedError("Regularization loss function not implemented")

    def backward_params(self):
        """
        Returns:
          Gradient of the L2 loss with respect to the regularized weights.
        """
        raise NotImplementedError("Regularized weights gradients function not implemented")

