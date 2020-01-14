from animation import plot
import numpy as np


def optimise(algorithm, function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, momentum
             , beta1, beta2, epsilon=1e-8, precision=0.1):
    if algorithm == "SGD":
        return sgd(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, precision)
    elif algorithm == "SGDM":
        return sgdm(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, momentum, precision)
    elif algorithm == "ADAM":
        return adam(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, beta1, beta2, epsilon,
                    precision)


def sgd(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, precision):
    """

    :param function: function to be optimised
    :param learning_rate: the value which denotes how big are the optimisation steps going to be
    :param initial_point: point from which the optimisimation is going to start
    :param num_of_iterations: number of iterations after which the optimsiation loop ends if the result is not reached
    :param is_to_be_plotted: boolean marks if the function will be plotted
    :param precision: the desired value of error
    :return final point in which function value is the minimum
    """
    domain_point = initial_point
    iterations = 0
    domain_point_vector = []
    gradient = function.__gradient__()(domain_point)

    while abs(gradient) >= precision or iterations < num_of_iterations:
        gradient = function.__gradient__()(domain_point)
        domain_point = domain_point - learning_rate * gradient
        domain_point_vector.append(domain_point)
        iterations += 1

    if is_to_be_plotted:
        plot(function, domain_point_vector, -50, 50)

    return domain_point


def sgdm(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, momentum, precision):
    """
    :param function: function to be optimised
    :param learning_rate: the value which denotes how big are the optimisation steps going to be
    :param initial_point: point from which the optimisimation is going to start
    :param num_of_iterations: number of iterations after which the optimsiation loop ends if the result is not reached
    :param is_to_be_plotted: boolean marks if the function will be plotted
    :param precision: the desired value of error
    :param momentum: the value which denotes how strongly will momentum influence the iteration result
    :return final point in which function value is the minimum
    """
    domain_point = initial_point
    iterations = 0
    domain_point_derivative = 0
    domain_point_vector = []
    gradient = function.__gradient__()(domain_point)

    while abs(gradient) >= precision or iterations < num_of_iterations:
        gradient = function.__gradient__()(domain_point)
        domain_point_derivative = momentum * domain_point_derivative - learning_rate * gradient
        domain_point = domain_point + domain_point_derivative
        domain_point_vector.append(domain_point)
        iterations += 1

    if is_to_be_plotted:
        plot(function, domain_point_vector, -50, 50)

    return domain_point


def adam(function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, beta1, beta2, epsilon, precision):
    """
    :param function: function to be optimised
    :param learning_rate: the value which denotes how big are the optimisation steps going to be
    :param initial_point: point from which the optimisimation is going to start
    :param num_of_iterations: number of iterations after which the optimsiation loop ends if the result is not reached
    :param is_to_be_plotted: boolean marks if the function will be plotted
    :param precision: the desired value of error
    :param beta1: exponential decay rate for the first moment
    :param beta2: exponential decay rate for the second moment
    :param epsilon: used as a divisor instead of zero
    :return final point in which function value is the minimum
    """
    domain_point = initial_point
    iterations = 0
    domain_point_vector = []
    gradient = function.__gradient__()(domain_point)
    first_moment_vector = 0
    second_moment_vector = 0

    while abs(gradient) >= precision or iterations < num_of_iterations:
        gradient = function.__gradient__()(domain_point)
        first_moment_vector = beta1 * first_moment_vector + (1 - beta1) * gradient
        second_moment_vector = beta2 * second_moment_vector + (1 - beta2) * np.power(gradient, 2)
        computational_first_moment_vector = first_moment_vector / (1 - np.power(beta1, iterations + 1))
        computational_second_moment_vector = second_moment_vector / (1 - np.power(beta2, iterations + 1))
        domain_point = domain_point - learning_rate * computational_first_moment_vector \
                       / (np.sqrt(computational_second_moment_vector) + epsilon)
        domain_point_vector.append(domain_point)
        iterations += 1

    if is_to_be_plotted:
        plot(function, domain_point_vector, -50, 50)

    return domain_point
