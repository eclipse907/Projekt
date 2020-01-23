import numpy as np
from LossFunction import LossFunction


class Loss(LossFunction):

    def forward(self):
        self.scores = self.model.scores2
        self.Y_oh = np.zeros(self.scores.shape)
        self.Y_oh[range(self.scores.shape[0]), self.Y_] = 1
        self.probs = self.stable_softmax(self.scores)
        logprobs = np.log(self.probs[range(self.model.N), self.Y_])  # N x 1
        regularized_loss = -(np.sum(logprobs) / self.model.N)  # skalar
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self):
        dL_ds = self.probs - self.Y_oh  # N x C
        return dL_ds
