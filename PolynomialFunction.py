from builtins import list, str, enumerate
from builtins import str
from itertools import zip_longest
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
        while function:
            term = function.pop(0)

            for letter in letters:
                if letter in term:
                    x, y = term.split(letter)
                    if x != '':
                        self.coefficients.append(x)
                    else:
                        self.coefficients.append(1)
                    if y != '':
                        powers.append(y)
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
        return self.__check_complete__(self.coefficients, powers)

    @staticmethod
    def __check_complete__(coefficients, powers):
        try:
            factor = 0
            for index in range(len(powers)):
                difference = powers[index] - powers[index+1]
                while difference > 1:
                    factor += 1
                    difference -= 1
                    coefficients.insert(index+1, 0)
        except:
            return coefficients

    def __call__(self, x):
        result = 0
        index = 0
        for coefficient in reversed(self.coefficients):
            result += coefficient * pow(x, index)
            index += 1
        return result

    def __max_degree__(self):
        return len(self.coefficients)

    def __add__(self, other):
        c1 = self.coefficients[::-1]
        c2 = other.coefficients[::-1]
        result = [sum(t) for t in zip_longest(c1, c2, fillvalue=0)]
        return PolynomialFunction(*result)

    def __sub__(self, other):
        c1 = self.coefficients[::-1]
        c2 = other.coefficients[::-1]
        result = [t1 - t2 for t1, t2 in zip_longest(c1, c2, fillvalue=0)]
        return PolynomialFunction(result)

    #we've got a bug here
    def __gradient__(self):
        fun = np.poly1d(self.coefficients)
        grad_fun = np.polyder(fun)
        return grad_fun

    def __random_set__(self, limit, number_of_samples, x):
        random_inputs = np.random.random_sample(number_of_samples)
        random_inputs = [element * limit - limit/2 for element in random_inputs]
        result = [self.__call__(element) for element in random_inputs]
        return result

