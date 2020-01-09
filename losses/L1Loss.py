from losses.LossFunction import *
import numpy as np


class L1Loss(LossFunction):

    def forward(self, Y_, Y):
        return 1 / np.size(Y_) * np.sum(np.abs(Y_ - Y))
