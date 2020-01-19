import numpy as np
import matplotlib.pyplot as plt
import data
from losses.HingeLoss import HingeLoss
from losses.L2Loss import L2Loss
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




def fcann2_train(X, Y_, param_niter = 100000):
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

    param_delta = 0.05
    param_lambda = np.exp(-3)
    regularizers = [L2Regularizer(W1, param_lambda, "l2reg_W1"), L2Regularizer(W2, param_lambda, "l2reg_W2")]
    loss = HingeLoss(Y_, regularizers)
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
    fcann2_setup_initial_params(X, Y_)
    W1, b1, W2, b2 = fcann2_train(X, Y_)
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argwhere(np.around(probs))[:, 1]

    acc, prec, conf_matrix = data.eval_perf_multi(Y, Y_)

    decfun = fcann2_decfun(W1, b1, W2, b2)
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, rect, offset=0)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    plt.show()
