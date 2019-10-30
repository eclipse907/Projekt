def optimise(algorithm, w):
    if algorithm is "SGD":
        sgd(w)
    elif algorithm is "SGDM":
        sgdm(w)
    elif algorithm is "ADAM":
        adam(w)


def sgd(w):
    ni = 0.1
    w.gradient()
    return


def sgdm(w):
    return


def adam(w):
    return
