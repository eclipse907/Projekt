import data
import numpy as np
import matplotlib.pyplot as plt
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
    b = np.zeros((1, C))
    # gradijentni spust (param_niter iteracija)
    #param_niter = 1000
    #param_delta = 0.3
    for i in range(param_niter):
        # eksponencirane klasifikacijske mjere
        # pri računanju softmaksa obratite pažnju
        # na odjeljak 4.1 udžbenika
        # (Deep Learning, Goodfellow et al)!
        scores = np.dot(X, W) + b  # N x C
        maxScores = np.amax(scores, axis=1)
        expscores = np.exp(scores - maxScores.reshape((N, 1))) # N x C

        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1)  # N x 1

        # logaritmirane vjerojatnosti razreda
        probs = expscores / sumexp.reshape((N, 1))  # N x C
        logprobs = np.log(probs[range(N), Y_])  # N x 1

        # gubitak
        loss = -(np.sum(logprobs) / N)  # scalar

        L2 = None
        if lambdaFactor is not None:
            L2 = L2Regularizer(W, lambdaFactor, "l2reg_" + str(i))
            l2RegLoss = L2.forward()
            loss += l2RegLoss

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        Yij = np.zeros((N,C))
        Yij[range(N), Y_] = 1
        dL_ds = probs - Yij  # N x C

        # gradijenti parametara
        grad_W = np.dot(np.transpose(dL_ds), X) / N  # C x D
        if lambdaFactor is not None:
            l2RegGradW = L2.backward_params()[0][1]
            grad_W += np.transpose(l2RegGradW)

        grad_b = np.sum(np.transpose(dL_ds), axis=1) / N  # C x 1

        # poboljšani parametri
        W += -param_delta * np.transpose(grad_W)
        b += -param_delta * np.transpose(grad_b)
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


if __name__=="__main__":
    np.random.seed(100)
    # get the training dataset
    X,Y_ = data.sample_gauss_2d(3, 200)

    # train the model
    W,b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = logreg_classify (X, W,b)
    Y = np.argwhere(np.around(probs))[:, 1]


    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print (accuracy, recall, precision)

    # graph the decision surface
    decfun = logreg_decfun(W,b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()