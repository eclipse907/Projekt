import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from onevariable_function import OnevariableFunction


def plot(function, values, min_x, max_x):
    x = np.linspace(min_x, max_x, 10000, endpoint=True)
    function_values = []

    for element in x:
        function_values.append(function.__call__(element))
    i = 1
    for w in values:
        plt.clf()
        plt.plot(x, function_values)
        plt.plot(w, function.__call__(w), 'ro')
        plt.draw()
        print("Iteration no. " + str(i) + "... Value of the function: " + str(function.__call__(w)))
        i += 1
        plt.pause(0.005)
