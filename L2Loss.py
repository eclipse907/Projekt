import numpy as np
from losses.LossFunction import LossFunction


class Loss(LossFunction):

    def forward(self):
        self.scores = self.model.scores2
        self.Y_oh = np.zeros(self.scores.shape)
        self.Y_oh[range(self.scores.shape[0]), self.Y_] = 1
        self.probs = self.stable_softmax(self.scores)
        regularized_loss = 0.5 * np.sum(np.square(self.probs - self.Y_oh))
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self):
        dL_ds = self.probs - self.Y_oh  # N x C
        # gradW = np.dot(dL_ds.T, scores)
        # gradb = np.sum(dL_ds.T, axis=1)
        # return [gradW, gradb]
        return dL_ds
