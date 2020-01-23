import numpy as np


def check_grad(grad, model, params, loss_class, X):
    # Numerički izačunati gradijent
    numGrad = compute_numerical_gradient(model, params, loss_class, X)

    # Usporedi
    brojnik = np.linalg.norm(grad - numGrad)
    nazivnik = np.linalg.norm(grad) + np.linalg.norm(numGrad)
    diff = brojnik / nazivnik
    if diff < params.tolerance:
        str = 'Dobar'
    else:
        str = 'Loš'
    return '{0} gradijent. Razlika = {1}'.format(str, diff)


def compute_numerical_gradient(model, params, loss_class, X):
    # Get the parameters (weights & biases)
    weights = model.get_params()
    h_vector = np.zeros(weights.shape)
    num_grad = np.zeros(weights.shape)

    for i in range(len(weights)):
        h_vector[i] = params.h

        model.set_params(weights + h_vector)
        model.forward_pass(X)
        f_plus = loss_class.forward()

        model.set_params(weights - h_vector)
        model.forward_pass(X)
        f_minus = loss_class.forward()

        # Izračunaj numerički gradijent
        num_grad[i] = (f_plus - f_minus) / (2 * params.h)
        h_vector[i] = 0

    # Resetiraj težine
    model.set_params(weights)

    return num_grad
