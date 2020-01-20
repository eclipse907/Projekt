import numpy as np
from losses.LossFunction import *


class Loss(LossFunction):

    def forward(self, scores):
        self.logits = scores
        self.Y_oh = np.zeros(scores.shape)
        self.Y_oh[range(scores.shape[0]), self.Y_] = 1
        self.probs = self.stable_softmax(scores)
        regularized_loss = 0.5 * np.sum(np.square(self.probs - self.Y_oh))
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self, scores):
        dL_ds = self.probs - self.Y_oh  # N x C
        # gradW = np.dot(dL_ds.T, scores)
        # gradb = np.sum(dL_ds.T, axis=1)
        # return [gradW, gradb]
        return dL_ds