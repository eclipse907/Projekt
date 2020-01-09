from abc import ABC, abstractmethod


class LossFunction(ABC):

    @abstractmethod
    def forward(self, Y_, Y):
        pass
