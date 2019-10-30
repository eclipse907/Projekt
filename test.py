import sys
from builtins import list, str, enumerate
from itertools import zip_longest
import numpy as np
import matplotlib.pyplot as plt
import optimisation as opt


class PolynomialFunction:

    coefficients = []

    def __init__(self, string):
        self.function = string
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.signs = ['+', '-', '*']

    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __parse__(self):
        for sign in self.signs:
            self.function = self.function.replace(sign, ' ')
            self.function = self.function.split()
            self.function.sort(reverse = True)
            powers = []
            while self.function:
                term = self.function.pop(0)

                for letter in self.letters:
                    if letter in term:
                        x, y = term.split(letter)
                        self.coefficients.append(int(x))
                        if y != '':
                            powers.append(int(y))
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

    def __check_complete__(self, coefficients, powers):
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
        result = [sum(t) for t in zip_longest(c1, c2, fillvalue = 0)]
        return PolynomialFunction(*result)

    def __sub__(self, other):
        c1 = self.coefficients[::-1]
        c2 = other.coefficients[::-1]
        result = [t1 - t2 for t1, t2 in zip_longest(c1, c2, fillvalue = 0)]
        return PolynomialFunction(result)

    def __gradient__(self):
        fun = np.poly1d(self.coefficients)
        grad_fun = np.polyder(fun)
        return grad_fun

    def __plot__(self, x1, x2):
        x = np.linspace(x1, x2, 50, endpoint=True)
        plt.plot(x, self.__call__(x))
        plt.show()

    def __plot__(self, x1, x2, dot):
        x = np.linspace(x1, x2, 50, endpoint=True)
        plt.plot(x, self.__call__(x))
        plt.plot(dot, self.__call__(dot), 'ro')
        plt.show()

    def __random_set__(self, limit, number_of_samples, x):
        random_inputs = np.random.random_sample(number_of_samples)
        random_inputs = [element * limit - limit/2 for element in random_inputs]
        result = [self.__call__(element) for element in random_inputs]
        return result


def main():
    function = PolynomialFunction([9, 6, 8])
    opt.optimise("SGDAnimation", function, 0.1, 10000, 10000)
    opt.optimise("SGDMAnimation", function, 0.1, 10000, 10000, 0.05)


if __name__ == '__main__':
    main()
