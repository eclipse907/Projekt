from losses.LossFunction import *
import numpy as np

class L1Loss(LossFunction):

    def calculateLoss(self, yTarget, yEstimate):
        return 1/np.size(yTarget) * np.sum(np.abs(yTarget - yEstimate))