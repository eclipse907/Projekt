import numpy as np
from losses.LossFunction import LossFunction

class Loss(LossFunction):

    def forward(self):
        self.scores = self.model.scores2
        self.Y_oh = np.zeros(self.model.scores2.shape)
        self.Y_oh[range(self.model.scores2.shape[0]), self.Y_] = 1
        self.true_class_scores = np.where(self.Y_oh == 1, self.model.scores2, 0).sum(axis=1, keepdims=True)
        scores_shape = self.model.scores2.shape
        self.margin = np.ones(scores_shape)
        loss_components = np.maximum(0, self.model.scores2 - self.true_class_scores + self.margin)
        loss_components[self.Y_oh == 1] = 0

        #self.probs = self.stable_softmax(self.scores)
        #self.probs = np.add(self.stable_softmax(self.scores), 1e-15)
        regularized_loss = np.sum(loss_components)
        if self.regularizers:
            for reg in self.regularizers:
                regularized_loss += reg.forward()
        return regularized_loss

    def backward_inputs(self):
        dL_ds = np.zeros(self.model.scores2.shape)
        class_losses = self.margin + self.model.scores2 - self.true_class_scores
        dL_ds[self.Y_oh == 1] -= np.where(self.Y_oh == 1, 0, class_losses > 0).sum(axis=1)
        dL_ds += np.where(self.Y_oh != 1, class_losses > 0, 0)
        return dL_ds