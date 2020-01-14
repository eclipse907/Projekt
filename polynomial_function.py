from builtins import str
import numpy as np


class PolynomialFunction:

    coefficients = []

    def __init__(self, val):
        if isinstance(val, str):
            self.function = str
            self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            self.__parse__(val, self.letters)

    def __parse__(self, function, letters):
        signs = ['+', '-', '*']

        for element in function:
            if element in signs:
                function = function.replace(element, ' ')
        function = function.split()
        function.sort(reverse=True)

        powers = []
        for term in function:
            for letter in letters:
                if letter in term:
                    x, y = term.split(letter)
                    if x != '':
                        self.coefficients.append(x)
                    else:
                        self.coefficients.append(1)
                    if y != '':
                        powers.append(y[1::])
                    else:
                        powers.append(1)
                else:
                    try:
                        temp = int(term)
                        self.coefficients.append(temp)
                        powers.append(0)
                        break
                    except:
                        pass
        self.coefficients = self.__check_coefficients__(self.coefficients, powers)
        return np.poly1d(self.coefficients)

    @staticmethod
    def __check_coefficients__(coefficients, powers):
        if len(coefficients) - 1 != powers[0]:
            while len(coefficients) < int(powers[0]) + 1:
                coefficients.append(0)
        return coefficients

    def __call__(self, x):
        return np.polyval(self.coefficients, x)

    def __gradient__(self):
        fun = np.poly1d(self.coefficients)
        grad_fun = np.polyder(fun)
        return grad_fun

    def __random_set__(self, limit, number_of_samples, x):
        random_inputs = np.random.random_sample(number_of_samples)
        random_inputs = [element * limit - limit / 2 for element in random_inputs]
        result = [self.__call__(element) for element in random_inputs]
        return result