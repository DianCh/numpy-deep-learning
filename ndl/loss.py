import numpy as np


class CrossEntropyLoss:
    def __init__(self, reduction="mean"):
        assert reduction in {
            "sum",
            "mean",
        }, "Invalid mode for loss reduction. Only 'sum' and 'mean' are supported."
        self.reduction = reduction
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
        # prob = np.expand_dims(x, axis=1) - np.expand_dims(x, axis=2)
        # prob = np.exp(prob)
        # prob = 1 / np.sum(prob, axis=2)
        x = x - np.max(x, axis=1, keepdims=True)
        prob = np.exp(x)
        prob = prob / np.sum(prob, axis=1, keepdims=True)

        loss = -np.sum(y * np.log(prob + 1e-8))
        if self.reduction == "mean":
            m, _ = x.shape
            loss /= m

        # save cache for back-propagation
        self.cache = prob, y

        return loss

    def backward(self):
        assert self.cache is not None, "Cannot backprop without forward first."
        prob, y = self.cache

        dX = prob - y
        if self.reduction == "mean":
            m, _ = prob.shape
            dX /= m

        # clear cache
        self.cache = None

        return dX
