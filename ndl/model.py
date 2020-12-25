import numpy as np


from ndl.layers import Conv2D, Linear, ReLU, Pool2D, Squeeze2D, Flatten


class Sequential:
    def __init__(self):
        self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dY):
        for layer in self.layers[::-1]:
            dY = layer.backward(dY)


class MultiLayerPerceptron(Sequential):
    def __init__(self, in_features, out_features, hidden):
        self.channels = [in_features, *hidden]

        self.layers = []
        for l in range(len(self.channels) - 1):
            linear = Linear(self.channels[l], self.channels[l + 1], bias=True)
            relu = ReLU()

            self.layers.append(linear)
            self.layers.append(relu)

        # last layer without non-linearity
        linear = Linear(self.channels[-1], out_features, bias=True)
        self.layers.append(linear)


class SimpleConvNet(Sequential):
    def __init__(self):
        self.layers = []

        self.layers.append(Conv2D(3, 8, 3, 1, 1))
        self.layers.append(ReLU())
        self.layers.append(Pool2D(2, 2))

        self.layers.append(Conv2D(8, 16, 3, 1, 1))
        self.layers.append(ReLU())
        self.layers.append(Pool2D(2, 2))

        self.layers.append(Conv2D(16, 32, 3, 1, 1))
        self.layers.append(ReLU())
        self.layers.append(Pool2D(2, 2, 0, "avg"))

        self.layers.append(Flatten())
        self.layers.append(Linear(4 * 4 * 32, 128))
        self.layers.append(ReLU())
        self.layers.append(Linear(128, 64))
        self.layers.append(ReLU())
        self.layers.append(Linear(64, 10))
