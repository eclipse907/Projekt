import numpy as np
import matplotlib.pyplot as plt
import data


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


class Params:

    def __init__(self, niter, delta):
        self.niter = niter
        self.delta = delta


def train(model, params):
    for i in range(params.niter):
        probs = model.forward_pass()
        loss_func(model, probs, i)
        grad_W1, grad_b1, grad_W2, grad_b2 = model.backward_pass(probs)
        model.W1 += -params.delta * np.transpose(grad_W1)
        model.b1 += -params.delta * grad_b1
        model.W2 += -params.delta * np.transpose(grad_W2)
        model.b2 += -params.delta * grad_b2


def fcann2_decfun(model):
    def classify(X):
        probs = model.forward_pass()
        return probs[:, 0]
    return classify


def loss_func(model, probs, i):
    logprobs = np.log(probs[range(model.N), model.Y_])  # N x 1
    loss = -(np.sum(logprobs) / model.N)  # skalar
    if i % 10 == 0:
        print("iteration {}: loss {}".format(i, loss))


if __name__ == "__main__":
    np.random.seed(100)
    model = Model(100, 2, 2)
    params = Params(50000, 0.17)
    model.random_dataset(5, 2, 20)
    train(model, params)
    probs = model.forward_pass()
    Y = np.argwhere(np.around(probs))[:, 1]
    decfun = fcann2_decfun(model)
    rect = (np.min(model.X, axis=0), np.max(model.X, axis=0))
    data.graph_surface(decfun, rect, offset=0)

    # graph the data points
    data.graph_data(model.X, model.Y_, Y, special=[])

    plt.show()
