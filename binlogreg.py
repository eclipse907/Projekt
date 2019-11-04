import numpy as np

import data


def sigmoid(s):
    return np.exp(s) / (1 + np.exp(s))


def binlogreg_train(X, Y_, param_niter=100, param_delta=0.1):
    """
        :param X: input data, np.ndarray NxD
        :param Y_: class labels, np.ndarray Nx1
        :param param_niter: hyperparameter, number of iterations
        :param param_delta: hyperparameter, delta for SGD

        :return: w, b - parameters for logistic regression
    """
    N, D = X.shape
    w = np.random.randn(D)
    b = 0
    for i in range(param_niter):
        scores = np.dot(X, w) + b
        probs = sigmoid(scores)
        loss = -1 / N * np.sum(np.log(probs))

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - Y_

        grad_w = 1 / N * np.dot(np.transpose(dL_dscores), X)
        grad_b = 1 / N * np.sum(dL_dscores)

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    """
        :param X: data, np.ndarray NxD
        :param w: logistic regression parameter, weights
        :param b: logistic regression parameter, bias

        :return: probs - probability for class c1
    """
    s = np.dot(X, w) + b
    probs = sigmoid(s)  # type: np.ndarray
    return probs


if __name__ == '__main__':
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = np.around(probs)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)
