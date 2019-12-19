import numpy as np
import matplotlib.pyplot as plt
import data
from regularizers.L2Regularizer import *

def fcann2_train(X, Y_,lambdaFactor=None, param_niter=100, param_delta=0.1):
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
    for i in range(param_niter):
        scores1 = np.dot(X, W1) + b1  # N x 5
        hiddenLayer1 = np.where(scores1 < 0, 0, scores1)  # N x 5
        scores2 = np.dot(hiddenLayer1, W2) + b2  # N x C
        maxScores2 = np.amax(scores2, axis=1) # 1 x N
        expscores2 = np.exp(scores2 - maxScores2.reshape((N, 1)))  # N x C
        sumexp2 = np.sum(expscores2, axis=1)  # 1 x N
        probs = expscores2 / sumexp2.reshape((N, 1))  # N x C
        logprobs = np.log(probs[range(N), Y_])  # N x 1
        loss = -(np.sum(logprobs) / N) # skalar

        L2_1 = None
        L2_2 = None
        if lambdaFactor is not None:
            L2_1 = L2Regularizer(W1, lambdaFactor, "l2reg_" + str(i))
            L2_2 = L2Regularizer(W2, lambdaFactor, "l2reg_" + str(i))
            l2RegLoss = L2_1.forward() + L2_2.forward()
            loss += l2RegLoss

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))
        Yij = np.zeros((N, C))
        Yij[range(N), Y_] = 1
        Gs2 = probs - Yij  # N x C
        grad_W2 = np.dot(np.transpose(Gs2), hiddenLayer1)  # C x 5
        if lambdaFactor is not None:
            l2RegGradW = L2_2.backward_params()[0][1]
            grad_W2 += np.transpose(l2RegGradW)
        grad_b2 = np.sum(np.transpose(Gs2), axis=1) # C x 1
        Gh1 = np.transpose(np.dot(W2, np.transpose(Gs2)))  # N x 5
        Gs1 = Gh1 # N x 5
        Gs1[hiddenLayer1 <= 0] = 0
        grad_W1 = np.dot(np.transpose(Gs1), X)  # 5 x D
        if lambdaFactor is not None:
            l2RegGradW = L2_1.backward_params()[0][1]
            grad_W1 += np.transpose(l2RegGradW)
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
    LAMBDA_FACTOR = np.exp(-3)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    # train/test split to 0.7/0.3
    mask = np.ones((int(6 * 10 * 0.3),), dtype=bool)
    mask = np.hstack((mask, np.zeros((int(6 * 10 * 0.7),), dtype=bool)))
    np.random.shuffle(mask)
    Xtest, Y_test = X[mask, :], Y_[mask]
    Xtrain, Y_train = X[np.logical_not(mask), :], Y_[np.logical_not(mask)]

    #W1, b1, W2, b2 = fcann2_train(Xtrain, Y_train, None)
    W1, b1, W2, b2 = fcann2_train(Xtrain, Y_train, LAMBDA_FACTOR)
    probs = fcann2_classify(Xtrain, W1, b1, W2, b2)
    #Y = np.argwhere(np.around(probs))[:, 1]
    Y = np.argmax(probs,axis=1)

    accuracy, recall, precision = data.eval_perf_multi(Y, Y_train)
    print("Train set:", accuracy, recall, precision)
    # AP = data.eval_AP(Y_test[probs.argsort()])



    probs = fcann2_classify(Xtest, W1, b1, W2, b2)
    #Y = np.argwhere(np.around(probs))[:, 1]
    Y = np.argmax(probs,axis=1)

    accuracy, recall, precision = data.eval_perf_multi(Y, Y_test)
    print("Test set:", accuracy, recall, precision)

    decfun = fcann2_decfun(W1, b1, W2, b2)
    rect = (np.min(Xtest, axis=0), np.max(Xtest, axis=0))
    data.graph_surface(decfun, rect, offset=0)

    # graph the data points
    data.graph_data(Xtest, Y_test, Y, special=[])

    plt.show()