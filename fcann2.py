import numpy as np
import matplotlib.pyplot as plt
import data
from losses.L2Loss import L2Loss
from regularizers.L2Regularizer import L2Regularizer


def fcann2_train(X, Y_):
    """
    Argumenti
      X: ulazni podaci, dimenzije NxD
      Y_: točni indeksi, dimenzije Nx1
    """
    N = X.shape[0]
    D = X.shape[1]
    C = max(Y_) + 1
    W1 = np.random.randn(D, 5)
    b1 = np.random.randn(1, 5)
    W2 = np.random.randn(5, C)
    b2 = np.random.randn(1, C)
    param_niter = 100000
    param_delta = 0.05
    param_lambda = 0
    regularizers = [L2Regularizer(W1, param_lambda, "l2reg_W1"), L2Regularizer(W2, param_lambda, "l2reg_W2")]
    loss = L2Loss(Y_, [regularizers])
    for i in range(param_niter):
        scores1 = np.dot(X, W1) + b1  # N x 5
        hiddenLayer1 = np.where(scores1 < 0, 0, scores1)  # N x 5
        scores2 = np.dot(hiddenLayer1, W2) + b2  # N x C
        # maxScores2 = np.amax(scores2, axis=1)  # 1 x N
        # expscores2 = np.exp(scores2 - maxScores2.reshape((N, 1)))  # N x C
        # sumexp2 = np.sum(expscores2, axis=1)  # 1 x N
        # probs = expscores2 / sumexp2.reshape((N, 1))  # N x C
        # logprobs = np.log(probs[range(N), Y_])  # N x 1
        # loss = -(np.sum(logprobs) / N)  # skalar
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))
        Yij = np.zeros((N, C))
        Yij[range(N), Y_] = 1
        Gs2 = probs - Yij  # N x C
        grad_W2 = np.dot(np.transpose(Gs2), hiddenLayer1)  # C x 5
        grad_b2 = np.sum(np.transpose(Gs2), axis=1)  # C x 1
        Gh1 = np.transpose(np.dot(W2, np.transpose(Gs2)))  # N x 5
        Gs1 = Gh1  # N x 5
        Gs1[Gs1 < 0] = 0
        grad_W1 = np.dot(np.transpose(Gs1), X)  # 5 x D
        grad_b1 = np.sum(np.transpose(Gs1), axis=1)  # 5 x 1
        W1 += -param_delta * np.transpose(grad_W1)
        b1 += -param_delta * grad_b1
        W2 += -param_delta * np.transpose(grad_W2)
        b2 += -param_delta * grad_b2
    return W1, b1, W2, b2


def fcann2_classify(X, W1, b1, W2, b2):
    N = X.shape[0]
    scores1 = np.dot(X, W1) + b1  # N x 5
    hiddenLayer1 = np.where(scores1 < 0, 0, scores1)  # N x 5
    scores2 = np.dot(hiddenLayer1, W2) + b2  # N x C
    maxScores2 = np.amax(scores2, axis=1)  # 1 x N
    expscores2 = np.exp(scores2 - maxScores2.reshape((N, 1)))  # N x C
    sumexp2 = np.sum(expscores2, axis=1)  # 1 x N
    probs = expscores2 / sumexp2.reshape((N, 1))  # N x C
    return probs


def fcann2_decfun(W1, b1, W2, b2):
    def classify(X):
        probs = fcann2_classify(X, W1, b1, W2, b2)
        return probs[:, 0]

    return classify


if __name__ == "__main__":
    np.random.seed(100)
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    W1, b1, W2, b2 = fcann2_train(X, Y_)
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argwhere(np.around(probs))[:, 1]
    decfun = fcann2_decfun(W1, b1, W2, b2)
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, rect, offset=0)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    plt.show()
