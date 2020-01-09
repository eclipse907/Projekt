import matplotlib.pyplot as plt
import numpy as np
import data
from binlogreg import *
from logreg import *

if __name__ == '__main__':

    np.random.seed(100)
    LAMBDA_FACTOR = np.exp(-3)
    n_samples = 400

    print("choose method:")
    print("0 bin logreg")
    print("1 logreg")
    trainMethod = int(input("enter method number: "))


    # w = np.random.randn(3)
    # l1SmoothLoss = L1SmoothLoss(w, 1, "l1smooth")
    # print("smooth l1 loss = ",l1SmoothLoss.forward())
    # print()
    # print("smooth l1 gradients = ", l1SmoothLoss.backward_params()[0][1])
    if trainMethod == 0:
        # get the training dataset
        X, Y_ = data.sample_gauss_2d(2, n_samples)

        # train/test split to 0.7/0.3
        mask = np.ones((int(2*n_samples*0.3),), dtype=bool)
        mask = np.hstack((mask, np.zeros((int(2*n_samples*0.7),), dtype=bool)))
        np.random.shuffle(mask)
        Xtest, Y_test = X[mask, :], Y_[mask]
        Xtrain, Y_train = X[np.logical_not(mask), :], Y_[np.logical_not(mask)]
        print(np.intersect1d(Xtrain, Xtest))

        # # train the model
        # BIN logreg
        w, b = binlogreg_train(Xtrain, Y_train, None)
        #w, b = binlogreg_train(Xtrain, Y_train, LAMBDA_FACTOR)

        # # evaluate the model on the train dataset
        probs = binlogreg_classify(Xtrain, w, b)
        Y = probs > 0.5
        # report performance
        accuracy, recall, precision = data.eval_perf_binary(Y, Y_train)
        AP = data.eval_AP(Y_train[probs.argsort()])
        print("Train set:", accuracy, recall, precision, AP)

        # # evaluate the model on the test dataset
        probs = binlogreg_classify(Xtest, w, b)
        Y = probs > 0.5
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

    elif trainMethod == 1:
        # get the training dataset
        X, Y_ = data.sample_gauss_2d(3, n_samples)

        # train/test split to 0.7/0.3
        mask = np.ones((int(3 * n_samples * 0.3),), dtype=bool)
        mask = np.hstack((mask, np.zeros((int(3 * n_samples * 0.7),), dtype=bool)))
        np.random.shuffle(mask)
        Xtest, Y_test = X[mask, :], Y_[mask]
        Xtrain, Y_train = X[np.logical_not(mask), :], Y_[np.logical_not(mask)]
        print(np.intersect1d(Xtrain, Xtest))

        # train the model
        W, b = logreg_train(Xtrain, Y_train, None, 10000, 0.1)
        #W, b = logreg_train(X, Y_, LAMBDA_FACTOR)

        # # evaluate the model on the train dataset
        probs = logreg_classify(Xtrain, W, b)
        Y = np.argmax(probs, axis=1)

        # report performance
        accuracy, recall, precision = data.eval_perf_multi(Y, Y_train)
        # AP = data.eval_AP(Y_train[probs.argsort()])
        print("Train set:", accuracy, recall, precision)#, AP)

        # # evaluate the model on the test dataset
        probs = logreg_classify(Xtest, W, b)
        Y = np.argmax(probs,axis=1)
        # report performance
        accuracy, recall, precision = data.eval_perf_multi(Y, Y_test)
        # AP = data.eval_AP(Y_test[probs.argsort()])
        print("Test set:", accuracy, recall, precision) #, AP)

        # graph the decision surface
        decfun = logreg_decfun(W, b)
        bbox = (np.min(Xtest, axis=0), np.max(Xtest, axis=0))
        data.graph_surface(decfun, bbox, offset=0.5)

        # graph the data points
        data.graph_data(Xtest, Y_test, Y, special=[])

        # show the plot
        plt.show()
    else:
        print("--end--")