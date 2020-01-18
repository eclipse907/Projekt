import numpy as np


class Optimizator:

    def __init__(self, algorithm, function, learning_rate, initial_point, num_of_iterations, momentum, beta1, beta2,
                 epsilon=1e-8):
        """
        :param algorithm: strings "SGD", "SGDM" and "ADAM" dentoe which algorithm is going to be used
        :param function: function to be optimised
        :param learning_rate: the value which denotes how big are the optimisation steps going to be
        :param initial_point: point from which the optimisimation is going to start
        :param num_of_iterations: number of iterations after which the optimsiation loop ends if the result is not reached
        :param momentum: the value which denotes how strongly will momentum influence the iteration result
        :param beta1: exponential decay rate for the first moment
        :param beta2: exponential decay rate for the second moment
        :param epsilon: used as a divisor instead of zero
        :return final point in which function value is the minimum
        """
        self.iterations = 0
        self.algorithm = algorithm
        self.function = function
        self.learning_rate = learning_rate
        self.initial_point = initial_point
        self.num_of_iterations = num_of_iterations
        self.momentum = momentum
        self.beta_1 = beta1
        self.beta_2 = beta2
        self.epsilon = epsilon
        self.domain_point = initial_point
        self.domain_point_derivative = 0
        self.first_moment_vector = 0
        self.second_moment_vector = 0

    def __call__(self):
        self.iterations += 1
        if self.algorithm == "SGD":
            return self.__sgd()
        elif self.algorithm == "SGDM":
            return self.__sgdm()
        elif self.algorithm == "ADAM":
            return self.__adam()

    def __sgd(self):
        self.domain_point = self.domain_point - self.learning_rate * self.function.__gradient__(self.domain_point)
        return self.function(self.domain_point)

    def __sgdm(self):
        gradient = self.function.__gradient__(self.domain_point)
        self.domain_point_derivative = self.momentum * self.domain_point_derivative - self.learning_rate * gradient
        self.domain_point = self.domain_point + self.domain_point_derivative
        return self.function(self.domain_point)

    def __adam(self):
        gradient = self.function.__gradient__(self.domain_point)
        self.first_moment_vector = self.beta_1 * self.first_moment_vector + (1 - self.beta_1) * gradient
        self.second_moment_vector = self.beta_2 * self.second_moment_vector + (1 - self.beta_2) * np.power(gradient, 2)
        computational_first_moment_vector = self.first_moment_vector / (1 - np.power(self.beta_1, self.iterations + 1))
        computational_second_moment_vector = \
            self.second_moment_vector / (1 - np.power(self.beta_2, self.iterations + 1))
        self.domain_point = self.domain_point - self.learning_rate * computational_first_moment_vector / \
                       (np.sqrt(computational_second_moment_vector) + self.epsilon)
        return self.function(self.domain_point)
