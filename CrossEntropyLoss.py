import numpy as np

def loss_func(model, probs, regularizer):
    logprobs = np.log(probs[range(model.N), model.Y_])  # N x 1
    loss = -(np.sum(logprobs) / model.N) + regularizer.forward()  # skalar
    return loss
