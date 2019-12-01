from LossFunction import *

class HingeLoss(LossFunction):

    def calculateLoss(self, yTarget, yEstimate):
        return max(0, 1 - yTarget * yEstimate)