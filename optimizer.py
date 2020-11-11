import numpy as np


class SGD:
    def __init__(self, model, lr, momentum):
        self.model = model
        self.lr = lr
        self.momentum = momentum

    def zero_grad(self):
        for layer in self.model.layers:
            if hasattr(layer, "dW") and hasattr(layer, "db"):
                layer.clear_gradients()

    def step(self, dY):
        for layer in self.model.layers[::-1]:
            dY = layer.backward(dY)
            if hasattr(layer, "dW") and hasattr(layer, "db"):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

