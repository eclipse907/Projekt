from losses.LossFunction import *
import numpy as np


class L2Loss(LossFunction):

    def forward(self, logits):
        self.logits = logits
        self.Y_oh = np.zeros(logits.shape)
        self.Y_oh[range(logits.shape[0]), self.Y_] = 1
        self.probs = self.stable_softmax(logits)
        regularized_loss = 0.5 * np.sum(np.square(self.probs - self.Y_oh))
        if self.regularizer:
            regularized_loss += sum((reg.forward() for reg in self.regularizer))
        return regularized_loss

    def backward(self):
        dL_ds = self.probs - self.Y_oh
        if self.regularizer:
            dL_ds += self.regularizer.backward_params()[0][1]
        return dL_ds
