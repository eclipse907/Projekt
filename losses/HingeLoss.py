from losses.LossFunction import *
import numpy as np


class HingeLoss(LossFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.margin = kwargs.get('margin', 1)

    def forward(self, scores):
        self.Y_oh = np.zeros(scores.shape)
        self.Y_oh[range(scores.shape[0]), self.Y_] = 1
        self.true_class_scores = np.where(self.Y_oh == 1, scores, 0).sum(axis=1, keepdims=True)
        loss_components = np.maximum(0, scores - self.true_class_scores + self.margin)
        loss_components[self.Y_oh == 1] = 0
        regularized_loss = np.sum(loss_components)
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self, scores):
        dL_ds = np.zeros(scores.shape)
        class_losses = self.margin + scores - self.true_class_scores
        dL_ds[self.Y_oh == 1] = - np.where(self.Y_oh == 1, 0, class_losses > 0).sum(axis=1)
        dL_ds += np.where(self.Y_oh != 1, class_losses > 0, 0)
        return dL_ds
