import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from onevariable_function import OnevariableFunction


def __function_wrapper(x, function):
    return [function(element) for element in x]


def plot(function, values, min_x, max_x):
    x = np.linspace(min_x, max_x, 10000, endpoint=True)
    y = __function_wrapper(x, function.__call__)

    i = 1
    for w in values:
        plt.clf()
        plt.plot(x, y)
        plt.plot(w, function.__call__(w), 'ro')
        if i == 1:
            plt.pause(1.5)
        plt.draw()
        print("Iteration no. " + str(i) + "... Value of the function: " + str(function.__call__(w)))
        i += 1
        plt.pause(0.005)
    plt.pause(5)
