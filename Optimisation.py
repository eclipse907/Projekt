from Animation import plot
import numpy as np

precision = 0.01


# add default parameters


def optimise(algorithm, function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, momentum, beta_1
             , beta_2, epsilon):
    if algorithm == "SGD":
        return sgd(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted)
    elif algorithm == "SGDM":
        return sgdm(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, momentum)
    elif algorithm == "ADAM":
        return adam(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, beta_1, beta_2
                    , epsilon)


def sgd(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted):
    """

    :param function: function to be optimised
    :param learning_rate: the value which denotes how big are the optimisation steps going to be
    :param initial_point: point from which the optimisimation is going to start
    :param num_of_iterations: number of iterations after which the optimsiation loop ends if the result is not reached
    :param is_to_be_plotted: boolean marks if the function will be plotted
    :return: final point in which function value is the minimum
    """
    w = initial_point
    iterations = 0
    values_w = []
    gradient = function.__gradient__()(w)
    while abs(gradient) >= precision or iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        w = w - learning_rate * gradient
        values_w.append(w)
        iterations += 1
    if is_to_be_plotted:
        plot(function, values_w, -50, 50)
    return w


def sgdm(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted=False, momentum=0.05):
    """

    :param function: function to be optimised
    :param learning_rate: the value which denotes how big are the optimisation steps going to be
    :param initial_point: point from which the optimisimation is going to start
    :param num_of_iterations: number of iterations after which the optimsiation loop ends if the result is not reached
    :param is_to_be_plotted: boolean marks if the function will be plotted
    :param momentum: the value which denotes how strongly will momentum influence the iteration result
    :return: final point in which function value is the minimum
    """
    w = initial_point
    iterations = 0
    dw = 0
    values_w = []
    gradient = function.__gradient__()(w)
    while abs(gradient) >= precision or iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        dw = momentum * dw - learning_rate * gradient
        w = w + dw
        values_w.append(w)
        iterations += 1
    if is_to_be_plotted:
        plot(function, values_w, -50, 50)
    return w


def adam(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, beta_1, beta_2, epsilon):
    """

    :param function: function to be optimised
    :param learning_rate: the value which denotes how big are the optimisation steps going to be
    :param initial_point: point from which the optimisimation is going to start
    :param num_of_iterations: number of iterations after which the optimsiation loop ends if the result is not reached
    :param is_to_be_plotted: boolean marks if the function will be plotted
    :param beta_1: exponential decay rate for the first moment
    :param beta_2: exponential decay rate for the second moment
    :param epsilon: used as a divisor instead of zero
    :return: final point in which function value is the minimum
    """
    w = initial_point
    iterations = 0
    values_w = []
    gradient = function.__gradient__()(w)
    m = 0
    v = 0
    while abs(gradient) >= precision or iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        m = beta_1 * m + (1 - beta_1) * gradient
        v = beta_2 * v + (1 - beta_2) * np.power(gradient, 2)
        m_hat = m / (1 - np.power(beta_1, iterations + 1))
        v_hat = v / (1 - np.power(beta_2, iterations + 1))
        w = w - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        values_w.append(w)
        iterations += 1
    if is_to_be_plotted:
        plot(function, values_w, -50, 50)
    return w
