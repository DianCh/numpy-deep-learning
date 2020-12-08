"""Implements layers that deal with reshaping the tensors."""
import numpy as np


class Squeeze2D:
    """Squeeze the (N, 1, 1, C) shape tensor into (N, C)."""

    def forward(self, x):
        """Squeeze at H, W dimensions."""
        return x.squeeze()

    def backward(self, dY):
        """Add back H, W dimensions."""
        return np.expand_dims(dY, axis=(1, 2))


class Flatten:
    """Flatten the (N, H, W, C) shape tensor into (N, CxHxW)."""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """Forward computation of reshaping."""
        m, h, w, c = x.shape

        # save cache for back-propagation
        self.cache = (m, h, w, c)

        return x.transpose((0, 3, 1, 2)).reshape((m, -1))

    def backward(self, dY):
        """Backward computation of gradients."""
        assert self.cache is not None, "Cannot backprop without forward first."
        m, h, w, c = self.cache

        # clear cache
        self.cache = None

        return dY.reshape((m, c, h, w)).transpose((0, 2, 3, 1))
