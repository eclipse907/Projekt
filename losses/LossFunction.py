from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):

    def __init__(self, Y_, regularizer):
        self.Y_ = Y_
        self.regularizer = regularizer

    @abstractmethod
    def forward(self, Y):
        pass

    @abstractmethod
    def backward(self, loss):
        pass
