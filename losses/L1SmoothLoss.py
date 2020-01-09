import numpy as np
from losses.LossFunction import *


class L1SmoothLoss(LossFunction):

    def forward(self, Y):
        """
         Returns:
          Scalar, smooth l1 loss.
        """
        self.Y = Y
        self.Y_oh = np.zeros(Y.shape)
        self.Y_oh[range(Y.shape[0]), self.Y_] = 1
        lossAllWeights = np.where(np.abs(Y - self.Y_oh) < 1,
                                  0.5 * np.square(Y - self.Y_oh),
                                  np.abs(Y - self.Y_oh) - 0.5)
        # print(lossAllWeights)
        regularized_loss = np.sum(lossAllWeights)
        if self.regularizer:
            regularized_loss += self.regularizer.forward()
        return [lossAllWeights, regularized_loss]

    def backward_params(self, loss):
        """
        Returns:
          Gradient of the smooth L1 loss with respect to the weights.
        """

        gradAllWeights = np.where(np.abs(self.Y - self.Y_oh) < 1,
                                  loss,
                                  np.sign(self.Y - self.Y_oh))
        # print(gradAllWeights)
        if self.regularizer:
            gradAllWeights += self.regularizer.backward_params()[0][1]
        return gradAllWeights
