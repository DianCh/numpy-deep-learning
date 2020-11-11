import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.cache = None

    def forward(self, x, y):
        """Calculate the cross-entropy loss for a batch.

        Args:
            x: (m, c) logits of each class
            y: (m, c) one-hot encoding for C-class labels
        
        Returns:
            loss: scalar cross-entropy loss
        """
        # calculate softmax probabilities
        prob = np.exp(x)
        prob = prob / np.sum(prob, axis=1, keepdims=True)

        loss = -np.sum(y * np.log(prob))

        # save cache for back-propagation
        self.cache = prob, y

        return loss

    def backward(self):
        assert self.cache is not None, "Cannot backprop without forward first."
        prob, y = self.cache

        dX = prob - y

        # clear cache
        self.cache = None

        return dX
