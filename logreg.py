import data
import numpy as np
import matplotlib.pyplot as plt

from losses.L2Loss import L2Loss
from regularizers.L2Regularizer import *

def logreg_train(X, Y_, lambdaFactor=None, param_niter=100, param_delta=0.1):
    '''
        Argumenti
          X:  podatci, np.array NxD
          Y_: indeksi razreda, np.array 1xN

        Povratne vrijednosti
          W, b: parametri logističke regresije
      '''
    N = X.shape[0]
    D = X.shape[1]
    C = max(Y_) + 1
    W = np.random.randn(D, C)
    print("before train: w =", W)
    b = np.zeros((1, C))
    # gradijentni spust (param_niter iteracija)
    #param_niter = 1000
    #param_delta = 0.3

    regularizers = [L2Regularizer(W, lambdaFactor, "l2reg")]
    loss = L2Loss(Y_, None)

    for i in range(param_niter):
        # eksponencirane klasifikacijske mjere
        # pri računanju softmaksa obratite pažnju
        # na odjeljak 4.1 udžbenika
        # (Deep Learning, Goodfellow et al)!
        scores = np.dot(X, W) + b  # N x C
        # maxScores = np.amax(scores, axis=1)
        # expscores = np.exp(scores - maxScores.reshape((N, 1))) # N x C
        #
        # # nazivnik sofmaksa
        # sumexp = np.sum(expscores, axis=1)  # N x 1
        #
        # # logaritmirane vjerojatnosti razreda
        # probs = expscores / sumexp.reshape((N, 1))  # N x C
        # logprobs = np.log(probs[range(N), Y_])  # N x 1
        #
        # # gubitak
        # loss = -(np.sum(logprobs) / N)  # scalar
        #
        # L2 = None
        # if lambdaFactor is not None:
        #     L2 = L2Regularizer(W, lambdaFactor, "l2reg_" + str(i))
        #     l2RegLoss = L2.forward()
        #     loss += l2RegLoss

        loss_sum = loss.forward(scores)

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss_sum))

        # # derivacije komponenata gubitka po mjerama
        # Yij = np.zeros((N, C))
        # Yij[range(N), Y_] = 1
        # dL_ds = probs - Yij  # N x C

        dL_ds = loss.backward()

        # gradijenti parametara
        # if lambdaFactor is not None:
        #     l2RegGradW = L2.backward_params()[0][1]
        #     dL_ds += np.transpose(l2RegGradW)

        grad_W = np.dot(np.transpose(dL_ds), X) / N  # C x D
        grad_b = np.sum(np.transpose(dL_ds), axis=1) / N  # C x 1

        # poboljšani parametri
        W += -param_delta * np.transpose(grad_W)
        b += -param_delta * np.transpose(grad_b)

    print("after train: w =", W)
    return W, b

def logreg_classify(X, W,b):
    N = X.shape[0]
    scores = np.dot(X, W) + b
    maxScores = np.amax(scores, axis=1)
    expscores = np.exp(scores - maxScores.reshape((N, 1)))
    sumexp = np.sum(expscores, axis=1)
    probs = expscores / sumexp.reshape((N, 1))
    return probs

def logreg_decfun(W,b):
    def classify(X):
      probs = logreg_classify(X, W,b)
      return probs[:, 0]
    return classify




