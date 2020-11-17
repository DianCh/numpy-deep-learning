import numpy as np


class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def zero_grad(self):
        for layer in self.model.layers:
            if hasattr(layer, "dW") and hasattr(layer, "db"):
                layer.clear_gradients()

    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, "dW") and hasattr(layer, "db"):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

