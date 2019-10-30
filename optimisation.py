import time


def optimise(algorithm, function, ni, w0, num_of_iterations=10000, alpha=0):
    if algorithm is "SGD":
        sgd(function, ni, w0, num_of_iterations)
    elif algorithm is "SGDM":
        sgdm(function, ni, w0, num_of_iterations, alpha)
    elif algorithm is "ADAM":
        adam(function, ni, w0, num_of_iterations)
    elif algorithm is "SGDAnimation":
        sgd_animation(function, ni, w0, num_of_iterations)
    elif algorithm is "SGDMAnimation":
        sgdm_animation(function, ni, w0, num_of_iterations, alpha)


def sgd(function, ni, w0, num_of_iterations):
    w = w0
    iterations = 0
    gradient = function.__gradient__()(w)
    while abs(gradient) > 0.0001 and iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        w = w - ni * gradient
        iterations += 1
    return w


def sgdm(function, ni, w0, num_of_iterations, alpha):
    w = w0
    iterations = 0
    gradient = function.__gradient__()(w)
    dw = 0
    while abs(gradient) > 0.0001 and iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        dw = alpha * dw - ni * gradient
        w = w + dw
        iterations += 1
    return w


def adam(function, ni, w0, num_of_iterations):
    return


def sgd_animation(function, ni, w0, num_of_iterations):
    w = w0
    iterations = 0
    gradient = function.__gradient__()(w)
    while abs(gradient) > 0.0001 and iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        w = w - ni * gradient
        iterations += 1
        if iterations % 10 == 0:
            function.__plot__(-10000, 10000, w)
            time.sleep(0.01)
    function.__plot__(-1, 1, w)
    return w


def sgdm_animation(function, ni, w0, num_of_iterations, alpha):
    w = w0
    iterations = 0
    gradient = function.__gradient__()(w)
    dw = 0
    while abs(gradient) > 0.0001 and iterations < num_of_iterations:
        gradient = function.__gradient__()(w)
        dw = alpha * dw - ni * gradient
        w = w + dw
        iterations += 1
        if iterations % 10 == 0:
            function.__plot__(-10000, 10000, w)
            time.sleep(0.01)
    function.__plot__(-1, 1, w)
    return w
