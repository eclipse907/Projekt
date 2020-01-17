import numpy as np
from losses.LossFunction import *


class L1SmoothLoss(LossFunction):

    def forward(self, scores):
        """
         Returns:
          Scalar, smooth l1 loss.
        """
        self.scores = scores
        self.Y_oh = np.zeros(scores.shape)
        self.Y_oh[range(scores.shape[0]), self.Y_] = 1
        lossAllWeights = np.where(np.abs(scores - self.Y_oh) < 1,
                                  0.5 * np.square(scores - self.Y_oh),
                                  np.abs(scores - self.Y_oh) - 0.5)
        # print(lossAllWeights)
        regularized_loss = np.sum(lossAllWeights)
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return [lossAllWeights, regularized_loss]

    def backward_inputs(self, previous_input):
        """
        Returns:
          Gradient of the smooth L1 loss with respect to the weights.
          :param previous_input:
        """

        gradAllWeights = np.where(np.abs(self.scores - self.Y_oh) < 1,
                                  self.scores - self.Y_oh,
                                  np.sign(self.scores - self.Y_oh))
        # print(gradAllWeights)
        if self.regularizers:
            gradAllWeights += sum((reg.backward_params()[0][1] for reg in self.regularizers))
        return gradAllWeights
