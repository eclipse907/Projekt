import numpy as np


def sigmoid(s):
    return np.exp(s) / (1 + np.exp(s))


def binlogreg_train(X, Y_, param_niter=100, param_delta=0.1):
    """
        :param X: input data, np.array NxD
        :param Y_: class labels, np.array Nx1
        :param param_niter: hyperparameter, number of iterations
        :param param_delta: hyperparameter, delta for SGD

        :return: w, b - parameters for logistical regression
    """
    N, D = X.shape
    w = np.random.randn(D)
    b = 0
    for i in range(param_niter):
        scores = np.dot(X, w) + b
        probs = sigmoid(scores)
        loss = -1/N * np.sum(np.log(probs))

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - Y_

        grad_w = 1/N * np.dot(np.transpose(dL_dscores), X)
        grad_b = 1/N * np.sum(dL_dscores)

        w += -param_delta * grad_w
        b += -param_delta * grad_b
