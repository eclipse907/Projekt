from losses.LossFunction import *
import numpy as np


class L1Loss(LossFunction):

    def forward(self, Y):
        self.logits = Y
        self.Y_oh = np.zeros(Y.shape)
        self.Y_oh[range(Y.shape[0]), self.Y_] = 1
        loss_components = np.abs(Y - self.Y_oh)
        regularized_loss = np.sum(loss_components)
        if self.regularizer:
            regularized_loss += sum((reg.forward() for reg in self.regularizer))
        return [loss_components, regularized_loss]

    def backward(self):
        grad = np.sign(self.logits - self.Y_oh)
        if self.regularizer:
            grad += self.regularizer.backward_params()[0][1]
        return grad
