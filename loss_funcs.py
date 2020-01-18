import numpy as np

def loss_func(model, probs, i):
    logprobs = np.log(probs[range(model.N), model.Y_])  # N x 1
    loss = -(np.sum(logprobs) / model.N)  # skalar
    if i % 10 == 0:
        print("iteration {}: loss {}".format(i, loss))
    return loss