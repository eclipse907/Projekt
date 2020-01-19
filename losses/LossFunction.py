from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):

    def __init__(self, Y_, regularizers):
        """
        :param Y_: array tocnih klasa, N x 1
        :param regularizers: lista regularizatora
        """
        self.Y_ = Y_
        self.regularizers = regularizers

    @abstractmethod
    def forward(self, scores):
        pass

    @abstractmethod
    def backward_inputs(self, scores):
        pass

    def backward_params(self):
        grads = []
        if self.regularizers:
            for reg in self.regularizers:
                grads += [reg.backward_params()]
        return grads

    def stable_softmax(self, x):
        N = x.shape[0]
        maxScores = np.amax(x, axis=1)
        expscores = np.exp(x - maxScores.reshape((N, 1)))  # N x C
        sumexp = np.sum(expscores, axis=1)  # N x 1
        probs = expscores / sumexp.reshape((N, 1))  # N x C
        return probs
