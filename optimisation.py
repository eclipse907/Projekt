from animation import plot
import numpy as np


def optimise(algorithm, function, ni, w0, num_of_iterations=10000, is_to_be_plotted=False, alpha=0, beta_1=0.9
             , beta_2=0.999, epsilon=10e-8):
    if algorithm is "SGD":
        sgd(function, ni, w0, num_of_iterations, is_to_be_plotted)
    elif algorithm is "SGDM":
        sgdm(function, ni, w0, num_of_iterations, is_to_be_plotted, alpha)
    elif algorithm is "ADAM":
        adam(function, ni, w0, num_of_iterations, is_to_be_plotted, beta_1, beta_2, epsilon)


def sgd(function, ni, w0, num_of_iterations, is_to_be_plotted):
    """

    :param function: function to optimise
    :param ni: learning rate
    :param w0: initial point
    :param num_of_iterations: number of iterations
    :param is_to_be_plotted: boolean marks what has to be plotted
    :return: final point in which function value is the minimum
    """
    w = w0
    iterations = 0
    values_w = []
    gradient = function.__gradient__()(w)
    while abs(gradient) >= 0.001 or iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        w = w - ni * gradient
        values_w.append(w)
        iterations += 1
    if is_to_be_plotted:
        plot(function, values_w, -50, 50)
    return w


def sgdm(function, ni, w0, num_of_iterations, is_to_be_plotted=False, alpha=0.05):
    """

    :param function: function to optimise
    :param ni: learning rate
    :param w0: initial point
    :param num_of_iterations: number of iterations
    :param is_to_be_plotted: boolean marks what has to be plotted
    :param alpha: momentum
    :return: final point in which function value is the minimum
    """
    w = w0
    iterations = 0
    dw = 0
    values_w = []
    gradient = function.__gradient__()(w)
    while abs(gradient) >= 0.001 or iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        dw = alpha * dw - ni * gradient
        w = w + dw
        values_w.append(w)
        iterations += 1
    if is_to_be_plotted:
        plot(function, values_w, -50, 50)
    return w


def adam(function, ni, w0, num_of_iterations, is_to_be_plotted, beta_1, beta_2, epsilon):
    """

    :param function: function to optimise
    :param ni: learning rate
    :param w0: initial point
    :param num_of_iterations: number of iterations
    :param is_to_be_plotted: boolean marks what has to be plotted
    :param beta_1: exponential decay rate for the first moment
    :param beta_2: exponential decay rate for the second moment
    :param epsilon: used as a divisor instead of zero
    :return: final point in which function value is the minimum
    """
    w = w0
    iterations = 0
    values_w = []
    gradient = function.__gradient__()(w)
    m = 0
    v = 0
    while abs(gradient) >= 0.001 or iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        m = beta_1 * m + (1 - beta_1) * gradient
        v = beta_2 * v + (1 - beta_2) * np.power(gradient, 2)
        m_hat = m / (1 - np.power(beta_1, iterations + 1))
        v_hat = v / (1 - np.power(beta_2, iterations + 1))
        w = w - ni * m_hat / (np.sqrt(v_hat) + epsilon)
        values_w.append(w)
        iterations += 1
    if is_to_be_plotted:
        plot(function, values_w, -50, 50)
    return w
