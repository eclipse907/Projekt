import argparse
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np

import data
import optimizator
from grad_check import check_grad


class Model:

    def __init__(self, N, D, C):
        self.N = N
        self.D = D
        self.C = C
        self.W1 = np.random.randn(D, 5)
        self.b1 = np.random.randn(1, 5)
        self.W2 = np.random.randn(5, C)
        self.b2 = np.random.randn(1, C)

    def random_dataset(self, ncomponents, nclasses, nsamples):
        self.X, self.Y_ = data.sample_gmm_2d(ncomponents, nclasses, nsamples)

    def init_dataset(self, X, Y_):
        self.X = X
        self.Y_ = Y_

    def forward_pass(self):
        self.scores1 = np.dot(self.X, self.W1) + self.b1  # N x 5
        self.hiddenLayer1 = np.where(self.scores1 < 0, 0, self.scores1)  # N x 5
        self.scores2 = np.dot(self.hiddenLayer1, self.W2) + self.b2  # N x C

    def backward_pass(self, Gs2):
        grad_W2 = np.dot(np.transpose(Gs2), self.hiddenLayer1) / self.N  # C x 5
        grad_b2 = np.sum(np.transpose(Gs2), axis=1) / self.N  # C x 1
        Gh1 = np.transpose(np.dot(self.W2, np.transpose(Gs2)))  # N x 5
        Gs1 = np.where(self.scores1 < 0, 0, Gh1)  # N x 5
        grad_W1 = np.dot(np.transpose(Gs1), self.X) / self.N  # 5 x D
        grad_b1 = np.sum(np.transpose(Gs1), axis=1) / self.N  # 5 x 1
        return grad_W1, grad_b1, grad_W2, grad_b2

    def copy(self):
        newModel = Model(self.N, self.D, self.C)
        newModel.W1, newModel.b1 = self.W1.copy(), self.b1.copy()
        newModel.W2, newModel.b2 = self.W2.copy(), self.b2.copy()
        newModel.X, newModel.Y_ = self.X.copy(), self.Y_.copy()
        # newModel.scores2 = self.scores2.copy()
        return newModel

    def get_params(self):
        return np.concatenate((self.W1.ravel(), self.W2.ravel(), self.b1.ravel(), self.b2.ravel()))

    def set_params(self, weights):
        W1_end = self.D*5
        self.W1 = np.reshape(weights[0:W1_end], (self.D, 5))

        W2_end = 5*self.C + W1_end
        self.W2 = np.reshape(weights[W1_end:W2_end], (5, self.C))

        b1_end = W2_end + 5
        self.b1 = np.reshape(weights[W2_end:b1_end], (1, 5))

        b2_end = b1_end + self.C
        self.b2 = np.reshape(weights[b1_end:b2_end], (1, self.C))


def train(model, params, lossClass, optimizationClass):
    for i in range(params.niter):
        model.forward_pass()
        loss = lossClass.forward()
        Gs2 = lossClass.backward_inputs()
        grad_W1, grad_b1, grad_W2, grad_b2 = model.backward_pass(Gs2)

        if i % 100 == 0:
            grad = np.concatenate((grad_W1.ravel(), grad_W2.ravel(), grad_b1.ravel(), grad_b2.ravel()))
            message = check_grad(grad, model, params, lossClass)
            print(message)

        if lossClass.regularizers:
            reg_grads = lossClass.backward_params()

            grad_W1 += reg_grads[0].T
            grad_W2 += reg_grads[1].T

        model.W1, model.W2 = optimizationClass(grad_W1, grad_W2)
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))
        model.b1 += -params.learning_rate_bias * grad_b1
        model.b2 += -params.learning_rate_bias * grad_b2


def fcann2_decfun(model):
    def classify(X):
        N = X.shape[0]
        scores1 = np.dot(X, model.W1) + model.b1  # N x 5
        hiddenLayer1 = np.where(scores1 < 0, 0, scores1)  # N x 5
        scores2 = np.dot(hiddenLayer1, model.W2) + model.b2  # N x C
        maxScores2 = np.amax(scores2, axis=1)  # 1 x N
        expscores2 = np.exp(scores2 - maxScores2.reshape((N, 1)))  # N x C
        sumexp2 = np.sum(expscores2, axis=1)  # 1 x N
        probs = expscores2 / sumexp2.reshape((N, 1))  # N x C
        Y = np.argmax(probs, axis=1)
        return Y
    return classify

def prepareXYSubtrainAndValidSets(X_train, Y_train):
    n_samples = X_train.shape[0]

    mask = np.ones((int(n_samples * paramsModule.valid_set_factor),), dtype=bool)
    mask = np.hstack((mask, np.zeros((int(n_samples * (1 - paramsModule.valid_set_factor)),), dtype=bool)))
    np.random.shuffle(mask)

    X_valid, Y_valid = X_train[mask, :], Y_train[mask]
    X_subtrain, Y_subtrain = X_train[np.logical_not(mask), :], Y_train[np.logical_not(mask)]

    return (X_subtrain, Y_subtrain), (X_valid, Y_valid)

# Algorithm 7.1
def findOptimalParams(model0, subtrain, valid, n, p):
    X_subtrain, Y_subtrain = subtrain[0], subtrain[1]
    X_valid, Y_valid = valid[0], valid[1]

    model = model0.copy()
    model.X, model.Y_ = X_subtrain, Y_subtrain
    model.N = X_subtrain.shape[0]
    i = 0
    j = 0
    v = np.inf
    model_star = model0.copy()
    i_star = i
    paramsModule.niter = n

    lossClass = lossModule.Loss(model, paramsModule, None)
    optimizationClass = optimizator.Optimizator(model, paramsModule, args.optimizer)

    while j < p:
        train(model, paramsModule, lossClass, optimizationClass)

        i = i + n

        classify = fcann2_decfun(model)
        Y = classify(X_valid)
        accuracy, pr, M = data.eval_perf_multi(Y, Y_valid)
        v_prime = 1. - accuracy

        if v_prime < v:
            j = 0
            model_star = model.copy()
            i_star = i
            v = v_prime
        else:
            j = j + 1

    return model_star, i_star, v


if __name__ == "__main__":
    np.random.seed(100)
    N = 100
    C = 2
    parser = argparse.ArgumentParser(description='Train a deep model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', default='parameters', help='Set the module with parameters')
    parser.add_argument('--loss', default='CrossEntropyLoss', help='Set the module with the loss')
    parser.add_argument('--optimizer', default='SGD', help='Set the optimizer')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='Use early stopping.')
    args = parser.parse_args()
    # print(args)
    model = Model(N, 2, C)
    model.random_dataset(5, 2, int(N / 5))
    paramsModule = import_module(args.params)
    lossModule = import_module(args.loss)
    # regularizerModule = import_module(args.regularizer)
    earlyStopping = args.early_stopping
    # regularizerClass = regularizerModule.Regularizer
    lossClass = lossModule.Loss(model, paramsModule, None)
    optimizationClass = optimizator.Optimizator(model, paramsModule, args.optimizer)
    # model.forward_pass()

    if earlyStopping:
        subtrain, valid = prepareXYSubtrainAndValidSets(model.X, model.Y_)
        # find optimal params by early stopping
        opt_model, opt_niter, opt_error = findOptimalParams(model, subtrain, valid, paramsModule.n_eval,
                                                            paramsModule.patience)
        model_new = model.copy()
        lossClass = lossModule.Loss(model_new, paramsModule, None)
        optimizationClass = optimizator.Optimizator(model_new, paramsModule, args.optimizer)
        train(model_new, paramsModule, lossClass, optimizationClass)
        model = model_new
    else:
        train(model, paramsModule, lossClass, optimizationClass)

    probs = lossClass.get_probs_from_scores(model.scores2)
    Y = np.argmax(probs, axis=1)
    decfun = fcann2_decfun(model)
    rect = (np.min(model.X, axis=0), np.max(model.X, axis=0))
    data.graph_surface(decfun, rect, offset=0)

    # graph the data points
    data.graph_data(model.X, model.Y_, Y, special=[])

    plt.show()
