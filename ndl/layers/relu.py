"""Implements the Rectified Linear Unit layer."""


class ReLU:
    """Standard ReLU layer."""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """Forward computation of ReLU activation."""
        # cap negative values to zero
        mask = x > 0.0
        out = x * mask

        # save cache for back-propagation
        self.cache = mask

        return out

    def backward(self, dY):
        """Backward computation of gradients."""
        assert self.cache is not None, "Cannot backprop without forward first."
        mask = self.cache

        # shut down gradients at negative positions
        dX = dY * mask

        # clear cache
        self.cache = None

        return dX
