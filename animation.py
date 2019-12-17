import matplotlib.pyplot as plt
import numpy as np


def plot(function, values, min_x, max_x):
    x = np.linspace(min_x, max_x, 50, endpoint=True)
    i = 1
    for w in values:
        plt.clf()
        plt.plot(x, function.__call__(x))
        plt.plot(w, function.__call__(w), 'ro')
        plt.draw()
        print("Iteration no. " + str(i) + "... Value of the function: " + str(function.__call__(w)))
        i += 1
        plt.pause(0.005)
