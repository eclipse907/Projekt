import numpy as np

class L1Regularizer():
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
    return [[self.weights, grad_weights], self.name]