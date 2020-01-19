from losses.LossFunction import *
import numpy as np


class L1Loss(LossFunction):

    def forward(self, scores):
        self.scores = scores
        self.Y_oh = np.zeros(scores.shape)
        self.Y_oh[range(scores.shape[0]), self.Y_] = 1
        loss_components = np.abs(scores - self.Y_oh)
        regularized_loss = np.sum(loss_components)
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self, scores):
        grad = np.sign(self.scores - self.Y_oh)
        return grad