"""Implements optimizers."""

import numpy as np


class SGD:
    """Standard Stochastic Gradient Descent optimizer with momentum."""

    def __init__(self, model, lr=0.01, momentum=0):
        self.layers = []
        self.velocities = []
        for layer in model.layers:
            if hasattr(layer, "dW") and hasattr(layer, "db"):
                self.layers.append(layer)
                self.velocities.append(
                    {
                        "dW": np.zeros((layer.dW.shape)),
                        "db": np.zeros((layer.db.shape)),
                    }
                )

        self.lr = lr
        self.momentum = momentum

    def zero_grad(self):
        """Clear out gradients of all registered learnable layers."""
        for layer in self.layers:
            if hasattr(layer, "dW") and hasattr(layer, "db"):
                layer.clear_gradients()

    def step(self):
        """Perform one step of descent of all registered learnable layers."""
        for layer, velocity in zip(self.layers, self.velocities):
            velocity["dW"] = self.momentum * velocity["dW"] + layer.dW
            velocity["db"] = self.momentum * velocity["db"] + layer.db
            layer.W -= self.lr * velocity["dW"]
            layer.b -= self.lr * velocity["db"]
