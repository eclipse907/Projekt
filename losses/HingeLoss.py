from losses.LossFunction import *
import numpy as np


class HingeLoss(LossFunction):

    def forward(self, scores):
        loss_components = np.max(0, np.ones(scores.shape) - np.multiply(self.Y_, scores))
        regularized_loss = np.sum(loss_components)
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return [loss_components, regularized_loss]

    def backward_inputs(self, previous_input):
        pass
