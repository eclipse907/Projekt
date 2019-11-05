from animation import plot

def optimise(algorithm, function, ni, w0, num_of_iterations=10000, alpha=0):
    if algorithm is "SGD":
        sgd(function, ni, w0, num_of_iterations)
    elif algorithm is "SGDM":
        sgdm(function, ni, w0, num_of_iterations, alpha)
    elif algorithm is "ADAM":
        adam(function, ni, w0, num_of_iterations)



def sgd(function, ni, w0, num_of_iterations):
    """

    :param function: function to optimise
    :param ni: learning rate
    :param w0: initial point
    :param num_of_iterations: number of iterations
    :return: final point in which function value is the minimum
    """
    w = w0
    iterations = 0
    gradient = function.__gradient__()(w)
    values_w = []
    while abs(gradient) >= 0.001 or iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        w = w - ni * gradient
        values_w.append(w)
        iterations += 1
    plot(function, values_w, -50, 50)
    return w


def sgdm(function, ni, w0, num_of_iterations, alpha):
    """

    :param function: function to optimise
    :param ni: learning rate
    :param w0: initial point
    :param num_of_iterations: number of iterations
    :param alpha: momentum
    :return: final point in which function value is the minimum
    """
    w = w0
    iterations = 0
    gradient = function.__gradient__()(w)
    dw = 0
    values_w = []
    while abs(gradient) >= 0.001 or iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        dw = alpha * dw - ni * gradient
        w = w + dw
        values_w.append(w)
        iterations += 1
    plot(function, values_w, -50, 50)
    return w


def adam(function, ni, w0, beta_1, beta_2, num_of_iterations):
    w = w0
    iterations = 0
    gradient = function.__gradient__()(w)
    #while abs(gradient) >= 0.001 or iterations < num_of_iterations:

