import numpy as np
from LossFunction import LossFunction

class Loss(LossFunction):

    def forward(self, y_):
        self.Y_oh = np.zeros(self.model.scores2.shape)
        self.Y_oh[range(self.model.scores2.shape[0]), y_] = 1
        self.true_class_scores = np.where(self.Y_oh == 1, self.model.scores2, 0).sum(axis=1, keepdims=True)
        loss_components = np.maximum(0, self.model.scores2 - self.true_class_scores + self.params.hinge_margin)
        loss_components[self.Y_oh == 1] = 0
        regularized_loss = np.sum(loss_components) / self.model.N
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self):
        dL_ds = np.zeros(self.model.scores2.shape)
        class_losses = self.params.hinge_margin + self.model.scores2 - self.true_class_scores
        dL_ds[self.Y_oh == 1] = - np.where(self.Y_oh == 1, 0, class_losses > 0).sum(axis=1)
        dL_ds += np.where(self.Y_oh != 1, class_losses > 0, 0)
        return dL_ds / self.model.N # TODO not to use model.N ...
