import numpy as np
from losses.LossFunction import *


class Loss(LossFunction):

    def forward(self, scores):
        """
         Returns:
          Scalar, smooth l1 loss.
        """
        self.scores = scores
        self.Y_oh = np.zeros(scores.shape)
        self.Y_oh[range(scores.shape[0]), self.Y_] = 1
        self.probs = self.stable_softmax(scores)
        # print(lossAllWeights)
        regularized_loss = np.sum(np.where(np.abs(self.probs - self.Y_oh) < 1,
                                  0.5 * np.square(self.probs - self.Y_oh),
                                  np.abs(self.probs - self.Y_oh) - 0.5))
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self, scores):
        """
        Returns:
          Gradient of the smooth L1 loss with respect to the weights.
          :param scores:
        """
        dL_ds = np.where(np.abs(self.probs - self.Y_oh) < 1,
                                  self.probs - self.Y_oh,
                                  np.sign(self.probs - self.Y_oh))
        return dL_ds
