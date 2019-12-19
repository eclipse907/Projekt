from abc import ABC, abstractmethod


class LossFunction(ABC):

    @abstractmethod
    def calculateLoss(self, yTarget, yEstimate):
        pass
