"""Implements the Sigmoid layer"""
import numpy as np


class Sigmoid:
    """Standard Sigmoid layer."""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """Forward computation of sigmoid."""
        out = 1 / (1 + np.exp(-x))

        # save cache for back-propagation
        self.cache = out

        return out

    def backward(self, dY):
        """Backward computation of gradients."""
        assert self.cache is not None, "Cannot backprop without forward first."
        y = self.cache

        dX = dY * y * (1 - y)

        # clear cache
        self.cache = None

        return dX
