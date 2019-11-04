import numpy as np

class L2Regularizer():
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
    # regularization term loss portion of L2 loss = lambda/2 * sum(weight**2)
    return self.weight_decay/2 * np.sum(np.square(self.weights))

  def backward_params(self):
    """
    Returns:
      Gradient of the L2 loss with respect to the regularized weights.
    """
    # TODO
    grad_weights = self.weight_decay * self.weights
    return [[self.weights, grad_weights], self.name]