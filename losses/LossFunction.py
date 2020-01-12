from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):

    def __init__(self, Y_, regularizer):
        """
        :param Y_: array tocnih klasa, N x 1
        :param regularizer: lista regularizatora
        """
        self.Y_ = Y_
        self.regularizer = regularizer

    @abstractmethod
    def forward(self, Y):
        pass

    @abstractmethod
    def backward(self):
        pass

    def stable_softmax(self, x):
        N = x.shape[0]
        maxScores = np.amax(x, axis=1)
        expscores = np.exp(x - maxScores.reshape((N, 1)))  # N x C
        sumexp = np.sum(expscores, axis=1)  # N x 1
        probs = expscores / sumexp.reshape((N, 1))  # N x C
        return probs
