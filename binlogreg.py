import numpy as np
from math import exp
from L2Regularizer import *

def sigmoid(s):
    return np.exp(s) / (1 + np.exp(s))


def binlogreg_train(X, Y_, w, lambdaFactor, param_niter=100, param_delta=0.1):
    """
        :param X: input data, np.array NxD
        :param Y_: class labels, np.array Nx1
        :param param_niter: hyperparameter, number of iterations
        :param param_delta: hyperparameter, delta for SGD

        :return: w, b - parameters for logistical regression
    """
    N, D = X.shape
    b = 0
    for i in range(param_niter):
        scores = np.dot(X, w) + b
        probs = sigmoid(scores)
        loss = -1/N * np.sum(np.log(probs))

        L2 = L2Regularizer(w, lambdaFactor, "l2reg_"+str(i))
        l2RegLoss = L2.forward()
        loss += l2RegLoss

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - Y_

        l2RegGradW = L2.backward_params()[0][1]

        grad_w = 1/N * np.dot(np.transpose(dL_dscores), X) + l2RegGradW
        grad_b = 1/N * np.sum(dL_dscores)

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return (w, b)




if __name__ == "__main__":
    np.random.seed(100)

    X, Y_ = np.empty([2, 3]), np.empty([2, 1])
    W = np.random.randn(X.shape[1], 1)

    lambdaFactor = exp(-3)

    wb = binlogreg_train(X, Y_, W, lambdaFactor)
    print(wb)

