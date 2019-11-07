import numpy as np
import matplotlib.pyplot as plt

import data

from L2Regularizer import *

def sigmoid(s):
    return np.exp(s) / (1 + np.exp(s))


def binlogreg_train(X, Y_, lambdaFactor, param_niter=100, param_delta=0.1):
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
        loss = -1 / N * np.sum(Y_ * np.log(probs) + (1 - Y_) * np.log(1 - probs))

        L2 = L2Regularizer(w, lambdaFactor, "l2reg_"+str(i))
        l2RegLoss = L2.forward()
        loss += l2RegLoss


        #if i % 10 == 0:
            #print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - Y_

        l2RegGradW = L2.backward_params()[0][1]

        grad_w = 1 / N * np.dot(np.transpose(dL_dscores), X) + np.transpose(l2RegGradW)
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


def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)

    return classify


def doTestSetCheck(tw, tb):
    print("---------------------\nTEST")
    testX, testY_ = data.sample_gauss_2d(2, 100)
    #print("...testX="+str(testX[:5]))
    print("...testY_=" + str(testY_))
    testProbs = binlogreg_classify(X, tw, tb)
    testY = testProbs > 0.5
    print("...testY=" + str(testY))
    # report performance
    Taccuracy, Trecall, Tprecision = data.eval_perf_binary(testY, testY_)
    AP = data.eval_AP(testY_[testProbs.argsort()])

    print("w="+str(w) + ", b="+str(b))
    print(Taccuracy, Trecall, Tprecision, AP)

if __name__ == '__main__':
    np.random.seed(10)
    LAMBDA_FACTOR = np.exp(-3)
    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)
    #print("...X=" + str(X[:5]))
    print("...Y_=" + str(Y_))

    # train the model
    w, b = binlogreg_train(X, Y_, LAMBDA_FACTOR)
    print("w=" + str(w) + ", b=" + str(b))
    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = probs > 0.5
    print("...Y=" + str(Y))
    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)

    doTestSetCheck(w, b)

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    # decfun = lambda X: binlogreg_classify(X, w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()




