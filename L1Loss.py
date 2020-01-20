import numpy as np
import LossFunction

class Loss(LossFunction):

    def forward(self):
        self.scores = self.model.scores2
        self.Y_oh = np.zeros(self.scores.shape)
        self.Y_oh[range(self.scores.shape[0]), self.Y_] = 1
        self.probs = self.stable_softmax(self.scores)
        regularized_loss = np.sum(np.abs(self.probs - self.Y_oh))
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self):
        dL_ds = np.sign(self.probs - self.Y_oh)
        return dL_ds
