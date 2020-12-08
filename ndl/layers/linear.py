"""Implements Linear layer."""
import numpy as np

from ndl.layers import Base


class Linear(Base):
    """Standard Linear layer."""

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.W = np.random.randn(in_features, out_features)
        self.b = np.zeros((out_features,))

        self.dW = np.zeros((in_features, out_features))
        self.db = np.zeros((out_features,))

        self.cache = None

        self.kaiming_uniform_init_weights()
        self.kaiming_uniform_init_biases()

    def forward(self, x):
        """Forward computation of linear multiplication."""
        # support 1-dimensional sample feature for now
        assert x.ndim == 2, "Only 1-dimensional feature is supported for now."

        # (m, C_prev) x (C_prev, C) + (C, )
        out = np.dot(x, self.W) + self.b

        # save cache for back-propagation
        self.cache = x

        return out

    def backward(self, dY):
        """Backward computation of gradients."""
        # clear existing gradients
        self.clear_gradients()

        assert self.cache is not None, "Cannot backprop without forward first."
        x = self.cache

        self.dW = np.dot(x.T, dY)
        if self.bias:
            self.db = np.sum(dY, axis=0)

        dX = np.dot(dY, self.W.T)

        # clear cache
        self.cache = None
        return dX

