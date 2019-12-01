from LossFunction import *
import numpy as np

class L2Loss(LossFunction):

    def calculateLoss(self, yTarget, yEstimate):
        return 1/np.size(yTarget) * np.sum(np.square(yTarget - yEstimate))