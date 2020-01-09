from losses.LossFunction import *


class HingeLoss(LossFunction):

    def forward(self, Y_, Y):
        return max(0, 1 - Y_ * Y)
