from losses.LossFunction import *
import numpy as np


class L2Loss(LossFunction):

    def forward(self, Y):
        self.Y = Y
        self.Y_oh = np.zeros(Y.shape)
        self.Y_oh[range(Y.shape[0]), self.Y_] = 1
        loss_components = 0.5 * np.square(Y - self.Y_oh)
        regularized_loss = np.sum(loss_components)
        if self.regularizer:
            regularized_loss += self.regularizer.forward()
        return [loss_components, regularized_loss]

    def backward(self, loss):
        grad = loss.clone()
        if self.regularizer:
            grad += self.regularizer.backward_params()[0][1]
        return grad
