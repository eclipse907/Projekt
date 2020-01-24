import numpy as np
from LossFunction import LossFunction


class Loss(LossFunction):

    def forward(self, y_):
        n = y_.shape[0]
        self.scores = self.model.scores2
        self.Y_oh = np.zeros(self.scores.shape)
        self.Y_oh[range(self.scores.shape[0]), y_] = 1
        self.probs = self.stable_softmax(self.scores)
        logprobs = np.log(self.probs[range(n), y_])  # N x 1
        regularized_loss = -(np.sum(logprobs) / n)  # skalar
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self):
        dL_ds = self.probs - self.Y_oh  # N x C
        return dL_ds
