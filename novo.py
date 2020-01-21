import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

class Random2DGaussian:

    d0min = 0
    d0max = 10
    d1min = 0
    d1max = 10
    scalecov = 5

    def __init__(self):
        dw0, dw1 = self.d0max - self.d0min, self.d1max - self.d1min
        mean = (self.d0min, self.d1min)
        mean += np.random.random_sample(2) * (dw0, dw1)
        eigvals = np.random.random_sample(2)
        eigvals *= (dw0 / self.scalecov, dw1 / self.scalecov)
        eigvals **= 2
        theta = np.random.random_sample() * np.pi * 2
        R = [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]]
        Sigma = np.dot(np.dot(np.transpose(R), np.diag(eigvals)), R)
        self.get_sample = lambda n: np.random.multivariate_normal(mean, Sigma, n)


class LogReg:

    tolerance = 10e-7
    h = 10e-5

    def __init__(self):

        self._niter = 2000
        self.learningRate = 0.01
        self.cost = []
        self.gradDiff = []
        self.dW = 0
        self.dB = 0
        self.novi_Y=[]

    def initLogReg(self, class_number,Y_):
        self.class_number = class_number
        self.W = np.random.rand(class_number,2)
        self.b = np.random.normal(0,1,class_number)
        self.training_number = len(Y_)

        self.novi_Y = np.zeros((self.training_number, self.class_number))
        for i in range(self.training_number):
            self.novi_Y[i][Y_[i]] = 1

    def train(self, X, Y_):
        self.initLogReg(max(Y_)+1,Y_)

        for i in range(self._niter):
            #Provjera ispravnosti gradijenta
            if (i%20==0):
                self.checkGrad(X)

            #Treniranje mreže
            prob=self.forward_backward_f(X)
            self.updateWeights()
        return self.forward_f(X), self.cost

    def forward_f(self,X):
        Y=np.dot(X, self.W.T) + self.b
        return self.stable_softmax(Y)

    def stable_softmax(self,z):
        exps = np.exp(z - np.max(z))
        return exps / np.array([np.sum(exps, axis=1)]).T

    def costFunction(self, prob, y):
        m = y.shape[0]
        cost = np.sum(y * np.log(prob))/m
        return -cost

    def forward_backward_f(self,X):
        prob = self.forward_f(X)
        cost = self.costFunction(prob, self.novi_Y)
        self.cost.append(cost)
        deriv = prob - self.novi_Y

        self.dW = np.dot(deriv.T, X) / self.training_number
        self.dB = np.sum(deriv.T, axis=1) / self.training_number
        return

    def updateWeights(self):
            self.W -= self.learningRate*self.dW
            self.b -= self.learningRate*self.dB

    def get_wb(self):
        return np.concatenate((self.W.ravel(), self.b.ravel()))

    def setWeights(self, weights):
        W_end=2*self.class_number
        self.W = np.reshape(weights[0:W_end], (self.class_number, 2))
        self.b = np.reshape(weights[W_end:], (self.class_number))

    def checkGrad(self, X):
        # Numerički izačunati gradijent
        numGrad = self.computeNumericalGradient(X)

        self.forward_backward_f(X)
        grad = np.concatenate((self.dW.ravel(), self.dB.ravel()))

        # Usporedi
        brojnik=np.linalg.norm(grad - numGrad)
        nazivnik=np.linalg.norm(grad)+np.linalg.norm(numGrad)
        diff = brojnik/nazivnik
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
            f_plus = self.costFunction(prob,self.novi_Y)

            self.setWeights(weights - h_vector)
            prob = self.forward_f(X)
            f_minus = self.costFunction(prob,self.novi_Y)
            # Izračunaj numerički gradijent
            numGrad[i] = (f_plus - f_minus) / (2 * self.h)
            h_vector[i] = 0
        # Resetiraj težine
        self.setWeights(weights)
        return numGrad




def data(nclasses, nsamples):
    Gs = []
    Ys = []
    for i in range(nclasses):
        Gs.append(Random2DGaussian())
        Ys.append(i)

    # sample the dataset
    X = np.vstack([G.get_sample(nsamples) for G in Gs])
    Y_ = np.hstack([[Y] * nsamples for Y in Ys])

    return X, Y_



def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_) + 1
    M = np.bincount(n * Y_ + Y, minlength=n * n).reshape(n, n)
    for i in range(n):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append((recall_i, precision_i))

    accuracy = np.trace(M) / np.sum(M)

    return accuracy, pr, M

def graph_data(X, Y_, Y):
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))

    colors[Y_ == 0] = [0.5,0.5,0.5]
    colors[Y_ == 1] = [0,0,0]

    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good],
                s=20, marker='o')

    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c='r',
                s=20, marker='s')


if __name__=="__main__":
    Y = []

    X,Y_= data(4,100)

    logreg=LogReg()
    Y,cost=logreg.train(X,Y_)



    Y=np.argmax(Y, axis=1)
    accuracy, pr, M=eval_perf_multi(Y, Y_)
    print(accuracy,pr,M)
    print(cost)

