import numpy as np
import matplotlib.pyplot as plt
import data
from losses.HingeLoss import HingeLoss
from losses.L1Loss import L1Loss
from losses.L1SmoothLoss import L1SmoothLoss
from losses.L2Loss import L2Loss
from regularizers.L1Regularizer import L1Regularizer
from regularizers.L2Regularizer import L2Regularizer

W1_initial, b1_initial, W2_initial, b2_initial = 0, 0, 0, 0

def fcann2_setup_initial_params(X, Y_):
    D = X.shape[1]
    C = max(Y_) + 1

    global W1_initial
    W1_initial = np.random.randn(D, 5)
    global b1_initial
    b1_initial = np.random.randn(1, 5)
    global W2_initial
    W2_initial = np.random.randn(5, C)
    global b2_initial
    b2_initial = np.random.randn(1, C)

    return W1_initial, b1_initial, W2_initial, b2_initial




def fcann2_train(X, Y_, param_niter = 50000):
    """
    Argumenti
      X: ulazni podaci, dimenzije NxD
      Y_: točni indeksi, dimenzije Nx1
    """
    global W1_initial, b1_initial, W2_initial, b2_initial
    N = X.shape[0]
    D = X.shape[1]
    C = max(Y_) + 1

    W1 = W1_initial
    b1 = b1_initial
    W2 = W2_initial
    b2 = b2_initial

    print("W1 = ", W1)
    print("W2 = ", W2)
    print("b1 = ", b1)
    print("b2 = ", b2)

    param_delta = 0.05
    param_lambda = np.exp(-3)
    #regularizers = []
    #regularizers = [L2Regularizer(W1, param_lambda, "l2reg_W1"), L2Regularizer(W2, param_lambda, "l2reg_W2")]
    regularizers = [L1Regularizer(W1, param_lambda, "l2reg_W1"), L1Regularizer(W2, param_lambda, "l2reg_W2")]
    loss = L1SmoothLoss(Y_, regularizers)
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

        loss_sum = loss.forward(scores2)

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss_sum))

        Gs2 = loss.backward_inputs(scores2)
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
        for grad in loss.backward_params():
            grad[0][0] += -param_delta * grad[0][1]

    print("W1 = ", W1)
    print("W2 = ", W2)
    print("b1 = ", b1)
    print("b2 = ", b2)

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

    K = 6
    C = 2
    N = 50
    X, Y_ = data.sample_gmm_2d(K, C, N)

    mask = np.ones((int(K * N * 0.3),), dtype=bool)
    mask = np.hstack((mask, np.zeros((int(K * N * 0.7),), dtype=bool)))
    np.random.shuffle(mask)
    Xtest, Y_test = X[mask, :], Y_[mask]
    Xtrain, Y_train = X[np.logical_not(mask), :], Y_[np.logical_not(mask)]

    fcann2_setup_initial_params(Xtrain, Y_train)
    W1, b1, W2, b2 = fcann2_train(Xtrain, Y_train)
    probs = fcann2_classify(Xtrain, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)

    acc, prec, conf_matrix = data.eval_perf_multi(Y, Y_train)
    print("Train set:", acc, prec, conf_matrix)


    probs = fcann2_classify(Xtest, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)

    acc, prec, conf_matrix = data.eval_perf_multi(Y, Y_test)
    print("Test set:", acc, prec, conf_matrix)

    # graph the decision surface
    decfun = fcann2_decfun(W1, b1, W2, b2)
    bbox = (np.min(Xtest, axis=0), np.max(Xtest, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(Xtest, Y_test, Y, special=[])


    plt.show()
