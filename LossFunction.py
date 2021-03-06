from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):

    def __init__(self, model, paramsModule, regularizerModule):
        """
        :param Y_: array tocnih klasa, N x 1
        :param regularizers: klasa regularizatora
                             ako je None nema regularizacije
        """
        self.model = model
        self.params = paramsModule
        self.regularizers = [] if regularizerModule is None else\
            [regularizerModule.Regularizer(model.W1, paramsModule.weight_decay), regularizerModule.Regularizer(model.W2, paramsModule.weight_decay)]

    @abstractmethod
    def forward(self, y_):
        pass

    @abstractmethod
    def backward_inputs(self):
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

    def get_probs_from_scores(self, scores, Y_):
        Y_oh = np.zeros(scores.shape)
        Y_oh[range(scores.shape[0]), Y_] = 1
        probs = self.stable_softmax(scores)

        return probs
