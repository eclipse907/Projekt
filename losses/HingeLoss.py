from losses.LossFunction import *
import numpy as np


class HingeLoss(LossFunction):

    def forward(self, Y):
        loss_components = np.max(0, np.ones(Y.shape) - np.multiply(self.Y_, Y))
        regularized_loss = np.sum(loss_components) + self.regularizer.forward()
        return [loss_components, regularized_loss]

    def backward(self, loss):
        pass
