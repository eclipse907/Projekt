from losses.LossFunction import *
import numpy as np


class HingeLoss(LossFunction):

    def forward(self, Y):
        loss_components = np.max(0, np.ones(Y.shape) - np.multiply(self.Y_, Y))
        regularized_loss = np.sum(loss_components)
        if self.regularizer:
            regularized_loss += sum((reg.forward() for reg in self.regularizer))
        return [loss_components, regularized_loss]

    def backward(self):
        pass
