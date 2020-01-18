import numpy as np
import matplotlib.pyplot as plt
import data
from importlib import import_module

class Model:

    def __init__(self, N, D, C):
        self.N = N
        self.D = D
        self.C = C
        self.W1 = np.random.randn(D, 5)
        self.b1 = np.random.randn(1, 5)
        self.W2 = np.random.randn(5, C)
        self.b2 = np.random.randn(1, C)

    def random_dataset(self, ncomponents, nclasses, nsamples):
        self.X, self.Y_ = data.sample_gmm_2d(ncomponents, nclasses, nsamples)

    def init_dataset(self, X, Y_):
        self.X = X
        self.Y = Y

    def forward_pass(self):
        self.scores1 = np.dot(self.X, self.W1) + self.b1  # N x 5
        self.hiddenLayer1 = np.where(self.scores1 < 0, 0, self.scores1)  # N x 5
        scores2 = np.dot(self.hiddenLayer1, self.W2) + self.b2  # N x C
        maxScores2 = np.amax(scores2, axis=1)  # 1 x N
        expscores2 = np.exp(scores2 - maxScores2.reshape((self.N, 1)))  # N x C
        sumexp2 = np.sum(expscores2, axis=1)  # 1 x N
        probs = expscores2 / sumexp2.reshape((self.N, 1))  # N x C
        return probs

    def backward_pass(self, probs):
        Yij = np.zeros((self.N, self.C))
        Yij[range(self.N), self.Y_] = 1
        Gs2 = probs - Yij  # N x C
        grad_W2 = np.dot(np.transpose(Gs2), self.hiddenLayer1) / self.N  # C x 5
        grad_b2 = np.sum(np.transpose(Gs2), axis=1) / self.N  # C x 1
        Gh1 = np.transpose(np.dot(self.W2, np.transpose(Gs2)))  # N x 5
        Gs1 = np.where(self.scores1 < 0, 0, Gh1)  # N x 5
        grad_W1 = np.dot(np.transpose(Gs1), self.X) / self.N  # 5 x D
        grad_b1 = np.sum(np.transpose(Gs1), axis=1) / self.N  # 5 x 1
        return grad_W1, grad_b1, grad_W2, grad_b2


def train(model, params, loss):
    for i in range(params.niter):
        probs = model.forward_pass()
        loss.loss_func(model, probs, i)
        grad_W1, grad_b1, grad_W2, grad_b2 = model.backward_pass(probs)
        model.W1 += -params.delta * np.transpose(grad_W1)
        model.b1 += -params.delta * grad_b1
        model.W2 += -params.delta * np.transpose(grad_W2)
        model.b2 += -params.delta * grad_b2


def fcann2_decfun(model):
    def classify(X):
        N = X.shape[0]
        scores1 = np.dot(X, model.W1) + model.b1  # N x 5
        hiddenLayer1 = np.where(scores1 < 0, 0, scores1)  # N x 5
        scores2 = np.dot(hiddenLayer1, model.W2) + model.b2  # N x C
        maxScores2 = np.amax(scores2, axis=1)  # 1 x N
        expscores2 = np.exp(scores2 - maxScores2.reshape((N, 1)))  # N x C
        sumexp2 = np.sum(expscores2, axis=1)  # 1 x N
        probs = expscores2 / sumexp2.reshape((N, 1))  # N x C
        return probs[:, 0]
    return classify


if __name__ == "__main__":
    np.random.seed(100)
    N = int(input("Unesite broj podataka:"))
    C = int(input("Unesite broj razreda:"))
    model = Model(N, 2, C)
    model.random_dataset(5, 2, int(N / 5))
    name = input("Unesite ime modula s parametrima:")
    paramsModule = import_module(name)
    name = input("Unesite ime modula sa funkcijom gubitka:")
    lossModule = import_module(name)
    train(model, paramsModule, lossModule)
    probs = model.forward_pass()
    Y = np.argwhere(np.around(probs))[:, 1]
    decfun = fcann2_decfun(model)
    rect = (np.min(model.X, axis=0), np.max(model.X, axis=0))
    data.graph_surface(decfun, rect, offset=0)

    # graph the data points
    data.graph_data(model.X, model.Y_, Y, special=[])

    plt.show()
