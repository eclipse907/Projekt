from losses.LossFunction import *
import numpy as np


class L1Loss(LossFunction):

    def forward(self, scores):
        self.scores = scores
        self.Y_oh = np.zeros(scores.shape)
        self.Y_oh[range(scores.shape[0]), self.Y_] = 1
        self.probs = self.stable_softmax(scores)
        regularized_loss = np.sum(np.abs(scores - self.Y_oh))
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self, scores):
        dL_ds = np.sign(self.scores - self.Y_oh)
        return dL_ds
