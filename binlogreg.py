import matplotlib.pyplot as plt

import data

from regularizers.L2Regularizer import *

def sigmoid(s):
    return np.exp(s) / (1 + np.exp(s))


def binlogreg_train(X, Y_, lambdaFactor=None, param_niter=100, param_delta=0.1):
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
        loss = -1 / N * np.sum(np.log(Y_ * probs + (1 - Y_) * (1 - probs)))

        if lambdaFactor is not None:
            L2 = L2Regularizer(w, lambdaFactor, "l2reg_"+str(i))
            l2RegLoss = L2.forward()
            loss += l2RegLoss

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - Y_

        if lambdaFactor is not None:
            l2RegGradW = L2.backward_params()[0][1]

        grad_w = 1 / N * np.dot(np.transpose(dL_dscores), X)
        if lambdaFactor is not None:
            grad_w += np.transpose(l2RegGradW)
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

if __name__ == '__main__':
    np.random.seed(10)

    # w = np.random.randn(3)
    # l1SmoothLoss = L1SmoothLoss(w, 1, "l1smooth")
    # print("smooth l1 loss = ",l1SmoothLoss.forward())
    # print()
    # print("smooth l1 gradients = ", l1SmoothLoss.backward_params()[0][1])
    LAMBDA_FACTOR = np.exp(-1)
    # get the training dataset
    n_samples = 400
    X, Y_ = data.sample_gauss_2d(2, n_samples)
    mask = np.ones((int(2*n_samples*0.3),), dtype=bool)
    mask = np.hstack((mask, np.zeros((int(2*n_samples*0.7),), dtype=bool)))
    np.random.shuffle(mask)
    Xtest, Y_test = X[mask, :], Y_[mask]
    Xtrain, Y_train = X[np.logical_not(mask), :], Y_[np.logical_not(mask)]
    print(np.intersect1d(Xtrain, Xtest))
    #print("...X=" + str(X[:5]))
    # print("...Y_=" + str(Y_))

    # # train the model
    w, b = binlogreg_train(Xtrain, Y_train, LAMBDA_FACTOR)
    # print("w=" + str(w) + ", b=" + str(b))

    # # evaluate the model on the train dataset
    probs = binlogreg_classify(Xtrain, w, b)
    Y = probs > 0.5
    # print("...Y=" + str(Y))
    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_train)
    AP = data.eval_AP(Y_train[probs.argsort()])
    print("Train set:", accuracy, recall, precision, AP)

    # # evaluate the model on the test dataset
    probs = binlogreg_classify(Xtest, w, b)
    Y = probs > 0.5
    # print("...Y=" + str(Y))
    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_test)
    AP = data.eval_AP(Y_test[probs.argsort()])
    print("Test set:", accuracy, recall, precision, AP)

    # # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    # decfun = lambda X: binlogreg_classify(X, w, b)
    bbox = (np.min(Xtest, axis=0), np.max(Xtest, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(Xtest, Y_test, Y, special=[])

    # show the plot
    plt.show()




