import numpy as np
from fcann2 import *

# Algorithm 7.2
def earlyStopping(X_train, Y_train, n, p, valid_set_factor=0.3):
    inOutSets = prepareXYSubtrainAndValidSets(X_train, Y_train, valid_set_factor)    # return: inOutSets = (X_valid, X_subtrain), (Y_valid, Y_subtrain)
    inSet = inOutSets[0]
    outSet = inOutSets[1]

    opt_niter = None

    # set theta0
    W1, b1, W2, b2 = fcann2_setup_initial_params(X_train, Y_train)
    # find optimal params by early stopping
    opt_params, opt_niter, opt_error = findOptimalParams(W1, b1, W2, b2, inSet, outSet, n, p)

    # set theta to random values again
    W1, b1, W2, b2 = fcann2_setup_initial_params(X_train, Y_train)
    W1, b1, W2, b2 = fcann2_train(X_train, Y_train, opt_niter) if opt_niter else fcann2_train(X_train, Y_train)


    return (W1, b1, W2, b2), opt_niter, opt_error





def prepareXYSubtrainAndValidSets(X_train, Y_train, valid_set_factor=0.3):
    n_samples = X_train.shape[0]

    mask = np.ones((int(n_samples * valid_set_factor),), dtype=bool)
    mask = np.hstack((mask, np.zeros((int(n_samples * (1 - valid_set_factor)),), dtype=bool)))
    np.random.shuffle(mask)

    X_valid, Y_valid = X_train[mask, :], Y_train[mask]
    X_subtrain, Y_subtrain = X_train[np.logical_not(mask), :], Y_train[np.logical_not(mask)]

    return (X_valid, X_subtrain), (Y_valid, Y_subtrain)

# Algorithm 7.1
def findOptimalParams(W1, b1, W2, b2, inSet, outSet, n, p):
    X_valid, X_subtrain = inSet[0], inSet[1]
    Y_valid, Y_subtrain = outSet[0], outSet[1]

    i = 0
    j = 0
    v = np.inf
    W1_star, b1_star, W2_star, b2_star = W1.copy(), b1.copy(), W2.copy(), b2.copy()
    i_star = i

    while j < p:
        W1, b1, W2, b2 = fcann2_train(X_subtrain, Y_subtrain, n)

        i = i + n

        Y = fcann2_classify(X_valid, W1, b1, W2, b2)
        accuracy, pr, M = data.eval_perf_multi(Y, Y_valid)
        v_prime = 1 - accuracy

        if v_prime < v:
            j = 0
            W1_star, b1_star, W2_star, b2_star = W1.copy(), b1.copy(), W2.copy(), b2.copy()
            i_star = i
            v = v_prime
        else:
            j = j + 1

    return (W1_star, b1_star, W2_star, b2_star), i_star, v


