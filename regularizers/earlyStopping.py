import numpy as np
from fcann2 import *

# Algorithm 7.2
def earlyStopping(X_train, Y_train, n, p, subtrain_valid_split_factor = 0.7):
    inOutSets = prepareXYSubtrainAndValidSets(X_train, Y_train, subtrain_valid_split_factor)    # return: inOutSets = (X_valid, X_subtrain), (Y_valid, Y_subtrain)
    inSet = inOutSets[0]
    outSet = inOutSets[1]

    # set theta0
    W1, b1, W2, b2 = fcann2_setup_initial_params(X_train, Y_train)
    # find optimal params by early stopping
    opt_params, opt_niter, opt_error = findOptimalParams(W1, b1, W2, b2, inSet, outSet, n, p)


    # set theta to random values again - TODO: not needed because W1, b1, W2, b2 are saved (line 11)?

    W1, b1, W2, b2 = fcann2_train(X_train, Y_train, opt_niter)

    return (W1, b1, W2, b2), opt_niter





def prepareXYSubtrainAndValidSets(X_train, Y_train, subtrain_valid_split_factor):
    n_samples = np.shape(X_train)[0] * np.shape(X_train)[1]

    mask = np.ones((int(n_samples * (1 - subtrain_valid_split_factor)),), dtype=bool)
    mask = np.hstack((mask, np.zeros((int(n_samples * subtrain_valid_split_factor),), dtype=bool)))
    np.random.shuffle(mask)

    X_valid, X_subtrain = X_train[mask, :], X_train[mask]
    Y_valid, Y_subtrain = Y_train[np.logical_not(mask), :], Y_train[np.logical_not(mask)]

    return (X_valid, X_subtrain), (Y_valid, Y_subtrain)

# Algorithm 7.1
def findOptimalParams(W1, b1, W2, b2, inSet, outSet, n, p):
    X_valid, X_subtrain = inSet[0], inSet[1]
    Y_valid, Y_subtrain = outSet[0], outSet[1]

    i = 0
    j = 0
    v = np.inf
    W1_star, b1_star, W2_star, b2_star = W1, b1, W2, b2
    i_star = i

    while j < p:
        W1, b1, W2, b2 = fcann2_train(X_subtrain, Y_subtrain, n)

        i = i + n

        #TODO
        v_prime = ValidationSetError(X_valid, Y_valid)
        if v_prime < v:
            j = 0
            W1_star, b1_star, W2_star, b2_star = W1, b1, W2, b2
            i_star = i
            v = v_prime
        else:
            j = j + 1

    return (W1_star, b1_star, W2_star, b2_star), i_star, v


