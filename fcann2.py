import numpy as np
import data
from importlib import import_module
from grad_check import check_grad
import optimizator
import argparse
import tensorflow as tf


class Model:

    def __init__(self, N, D, C, H):
        self.N = N
        self.D = D
        self.C = C
        self.H = H # Number of neurons in hidden layers
        self.W1 = np.random.randn(D, H)
        self.b1 = np.random.randn(1, H)
        self.W2 = np.random.randn(H, C)
        self.b2 = np.random.randn(1, C)

    def forward_pass(self, X):
        self.scores1 = np.dot(X, self.W1) + self.b1  # N x 5
        self.hiddenLayer1 = np.where(self.scores1 < 0, 0, self.scores1)  # N x 5
        self.scores2 = np.dot(self.hiddenLayer1, self.W2) + self.b2  # N x C

    def backward_pass(self, Gs2, X):
        grad_W2 = np.dot(np.transpose(Gs2), self.hiddenLayer1) / self.N  # C x 5
        grad_b2 = np.sum(np.transpose(Gs2), axis=1) / self.N  # C x 1
        Gh1 = np.transpose(np.dot(self.W2, np.transpose(Gs2)))  # N x 5
        Gs1 = np.where(self.scores1 < 0, 0, Gh1)  # N x 5
        grad_W1 = np.dot(np.transpose(Gs1), X) / self.N  # 5 x D
        grad_b1 = np.sum(np.transpose(Gs1), axis=1) / self.N  # 5 x 1
        return grad_W1, grad_b1, grad_W2, grad_b2

    def copy(self):
        newModel = Model(self.N, self.D, self.C)
        newModel.W1, newModel.b1 = self.W1.copy(), self.b1.copy()
        newModel.W2, newModel.b2 = self.W2.copy(), self.b2.copy()
        newModel.scores2 = self.scores2.copy()
        return newModel

    def get_params(self):
        return np.concatenate((self.W1.ravel(), self.W2.ravel(), self.b1.ravel(), self.b2.ravel()))

    def set_params(self, weights):
        W1_end = self.D*self.H
        self.W1 = np.reshape(weights[0:W1_end], (self.D, self.H))

        W2_end = self.H*self.C + W1_end
        self.W2 = np.reshape(weights[W1_end:W2_end], (self.H, self.C))

        b1_end = W2_end + self.H
        self.b1 = np.reshape(weights[W2_end:b1_end], (1, self.H))

        b2_end = b1_end + self.C
        self.b2 = np.reshape(weights[b1_end:b2_end], (1, self.C))


def train(model, params, lossClass, optimizationClass, X, Y_):
    for i in range(params.niter):
        if i % 100 == 0:
            x_subset = X[:100,:] # mini batch just for grad check not to go through whole dataset
            y_subset = Y_[:100]
            model.forward_pass(x_subset)
            loss = lossClass.forward(y_subset)
            Gs2 = lossClass.backward_inputs()
            grad_W1, grad_b1, grad_W2, grad_b2 = model.backward_pass(Gs2, x_subset)
            grad = np.concatenate((grad_W1.ravel(), grad_W2.ravel(), grad_b1.ravel(), grad_b2.ravel()))
            message = check_grad(grad, model, params, lossClass, x_subset, y_subset)
            print(message)

        model.forward_pass(X)
        loss = lossClass.forward(Y_)
        Gs2 = lossClass.backward_inputs()
        grad_W1, grad_b1, grad_W2, grad_b2 = model.backward_pass(Gs2, X)

        if lossClass.regularizers:
            reg_grads = lossClass.backward_params()

            grad_W1 += reg_grads[0].T
            grad_W2 += reg_grads[1].T

        model.W1, model.W2 = optimizationClass(grad_W1, grad_W2)
        if i % 10 == 0:
            print("Iteration {}: loss {}".format(i, loss))
        model.b1 += -params.learning_rate_bias * grad_b1
        model.b2 += -params.learning_rate_bias * grad_b2


def classify(X, model):
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


def prepareXYSubtrainAndValidSets(X_train, Y_train):
    n_samples = X_train.shape[0]

    mask = np.ones((int(n_samples * paramsModule.valid_set_factor),), dtype=bool)
    mask = np.hstack((mask, np.zeros((int(n_samples * (1 - paramsModule.valid_set_factor)),), dtype=bool)))
    np.random.shuffle(mask)

    X_valid, Y_valid = X_train[mask, :], Y_train[mask]
    X_subtrain, Y_subtrain = X_train[np.logical_not(mask), :], Y_train[np.logical_not(mask)]

    return (X_subtrain, Y_subtrain), (X_valid, Y_valid)


# Algorithm 7.1
def findOptimalParams(model0, regularizerModule, paramsModule, subtrain, valid, n, p):
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

    lossClass = lossModule.Loss(model, paramsModule, regularizerModule, Y_subtrain)
    optimizationClass = optimizator.Optimizator(model, paramsModule, args.optimizer)

    while j < p:
        train(model, paramsModule, lossClass, optimizationClass, X_subtrain)

        i = i + n

        Y = classify(X_valid, model)
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
    # np.random.seed(100)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    N = x_train.shape[0]
    D = x_train.shape[1] * x_train.shape[2]
    C = np.max(y_train) + 1
    x_train = np.reshape(x_train, (N, D))
    parser = argparse.ArgumentParser(description='Train a deep model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', default='parameters', help='Set the module with parameters')
    parser.add_argument('--loss', default='CrossEntropyLoss', help='Set the module with the loss')
    parser.add_argument('--regularizer', default='L2Regularizer', help='Set the regularizer')
    parser.add_argument('--optimizer', default='SGD', help='Set the optimizer')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='Use early stopping.')
    args = parser.parse_args()
    paramsModule = import_module(args.params)
    lossModule = import_module(args.loss)
    regularizerModule = import_module(args.regularizer)
    earlyStopping = args.early_stopping

    model = Model(N, D, C, paramsModule.hidden_layer_neurons)
    lossClass = lossModule.Loss(model, paramsModule, regularizerModule)
    optimizationClass = optimizator.Optimizator(model, paramsModule, args.optimizer)
    model.forward_pass(x_train)

    if earlyStopping:
        subtrain, valid = prepareXYSubtrainAndValidSets(x_train, y_train)
        # find optimal params by early stopping
        opt_model, opt_niter, opt_error = findOptimalParams(model, regularizerModule, paramsModule, subtrain, valid, paramsModule.n_eval,
                                                            paramsModule.patience)
        model_new = model.copy()
        optimizationClass = optimizator.Optimizator(model_new, paramsModule, args.optimizer)
        print("-------------------*********", opt_niter, "-------------------*********")
        train(model_new, paramsModule, lossClass, optimizationClass, x_train, y_train)
        model = model_new
    else:
        train(model, paramsModule, lossClass, optimizationClass, x_train, y_train)

    print("W1 = ", model.W1)
    print("W2 = ", model.W2)
    probs = lossClass.get_probs_from_scores(model.scores2, y_train)
    Y = np.argmax(probs, axis=1)
    accuracy, recall, precision = data.eval_perf_multi(Y, y_train)
    print(accuracy, recall, precision)

    N_test = x_test.shape[0]
    D_test = x_test.shape[1] * x_test.shape[2]
    C_test = np.max(y_test) + 1
    x_test = np.reshape(x_test, (N_test, D_test))
    lossClass.Y_ = y_test
    model.forward_pass(x_test)
    probs = lossClass.get_probs_from_scores(model.scores2, y_test)
    Y = np.argmax(probs, axis=1)
    accuracy, recall, precision = data.eval_perf_multi(Y, y_test)
    print(accuracy, recall, precision)
