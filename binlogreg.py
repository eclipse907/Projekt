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
    print("before train: w =", w)
    b = 0
    for i in range(param_niter):
        scores = np.dot(X, w) + b
        probs = sigmoid(scores)
        loss = -1 / N * np.sum(np.log(Y_ * probs + (1 - Y_) * (1 - probs)))

        L2 = None
        if lambdaFactor is not None:
            L2 = L2Regularizer(w, lambdaFactor, "l2reg_"+str(i))
            l2RegLoss = L2.forward()
            loss += l2RegLoss

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - Y_

        grad_w = 1 / N * np.dot(np.transpose(dL_dscores), X)
        if lambdaFactor is not None:
            l2RegGradW = L2.backward_params()[0][1]
            grad_w += np.transpose(l2RegGradW)

        grad_b = 1 / N * np.sum(dL_dscores)

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    print("after train: w =", w)
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






