import numpy as np


def checkGrad(self, X):
    # Numerički izačunati gradijent
    numGrad = self.computeNumericalGradient(X)

    self.forward_backward_f(X)
    grad = np.concatenate((self.dW.ravel(), self.dB.ravel()))

    # Usporedi
    brojnik = np.linalg.norm(grad - numGrad)
    nazivnik = np.linalg.norm(grad) + np.linalg.norm(numGrad)
    diff = brojnik / nazivnik
    if diff < self.tolerance:
        str = 'Dobar'
    else:
        str = 'Loš'
    print('{0} gradijent. Razlika = {1}'.format(str, diff))
    self.gradDiff.append(diff)


def computeNumericalGradient(self, X):
    weights = self.get_wb()
    h_vector = np.zeros(weights.shape)
    numGrad = np.zeros(weights.shape)

    for i in range(len(weights)):
        h_vector[i] = self.h

        self.setWeights(weights + h_vector)
        prob = self.forward_f(X)
        f_plus = self.costFunction(prob, self.novi_Y)

        self.setWeights(weights - h_vector)
        prob = self.forward_f(X)
        f_minus = self.costFunction(prob, self.novi_Y)
        # Izračunaj numerički gradijent
        numGrad[i] = (f_plus - f_minus) / (2 * self.h)
        h_vector[i] = 0
    # Resetiraj težine
    self.setWeights(weights)
    return numGrad
