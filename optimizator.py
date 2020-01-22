import numpy as np


class Optimizator:

    def __init__(self, model, params, algorithm):
        self.algorithm = algorithm
        self.model = model
        self.params = params
        self.iterations = 0
        self.domain_point_derivative_1 = 0
        self.domain_point_derivative_2 = 0
        self.first_moment_vector_1 = 0
        self.first_moment_vector_2 = 0
        self.second_moment_vector = 0

    def __call__(self, grad_W1, grad_W2):
        self.iterations += 1
        if self.algorithm.upper() == "SGD":
            return self.__sgd(grad_W1, grad_W2)
        elif self.algorithm.upper() == "SGDM":
            return self.__sgdm(grad_W1, grad_W2)
        elif self.algorithm.upper() == "ADAM":
            return self.__adam(grad_W1, grad_W2)

    def __sgd(self, grad_W1, grad_W2):
        w1 = np.transpose(self.model.W1) - self.params.learning_rate_sgd * grad_W1
        w2 = np.transpose(self.model.W2) - self.params.learning_rate_sgd * grad_W2

        return w1, w2

    def __sgdm(self, grad_W1, grad_W2):
        self.domain_point_derivative_1 = self.params.momentum * self.domain_point_derivative_1 - self.params.learning_rate_sgdm * grad_W1
        self.domain_point_derivative_1 = self.params.momentum * self.domain_point_derivative_2 - self.params.learning_rate_sgdm * grad_W2

        w1 = np.transpose(self.model.W1) + self.domain_point_derivative_1
        w2 = np.transpose(self.model.W2) + self.domain_point_derivative_2

        return w1, w2

    def __adam(self, grad_W1, grad_W2):
        self.first_moment_vector_1 = self.params.beta_1 * self.first_moment_vector_1 + (1 - self.params.beta_1) * grad_W1
        self.first_moment_vector_2 = self.params.beta_1 * self.first_moment_vector_2 + (1 - self.params.beta_1) * grad_W2
        self.second_moment_vector_1 = self.params.beta_2 * self.second_moment_vector + (1 - self.params.beta_2) * np.power(grad_W1, 2)
        self.second_moment_vector_2 = self.params.beta_2 * self.second_moment_vector + (1 - self.params.beta_2) * np.power(grad_W2, 2)
        computational_first_moment_vector = self.first_moment_vector_1 / (1 - np.power(self.params.beta_1, self.iterations + 1))
        computational_second_moment_vector = \
            self.second_moment_vector / (1 - np.power(self.params.beta_2, self.iterations + 1))

        w1 = np.transpose(self.model.W1) - self.params.learning_rate * computational_first_moment_vector / \
                       (np.sqrt(computational_second_moment_vector) + self.params.epsilon)
        w2 = np.transpose(self.model.W2) - self.params.learning_rate * computational_first_moment_vector / \
                       (np.sqrt(computational_second_moment_vector) + self.params.epsilon)

        return w1, w2
