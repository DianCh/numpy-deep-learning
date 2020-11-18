import numpy as np


from ndl.layer import Conv2D, Linear, ReLU, Pool2D, Squeeze2D, Flatten


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

        self.layers.append(Conv2D(3, 6, 5))
        self.layers.append(ReLU())
        self.layers.append(Pool2D(2, 2))

        self.layers.append(Conv2D(6, 16, 5, 1, 0))
        self.layers.append(ReLU())
        self.layers.append(Pool2D(2, 2))

        self.layers.append(Flatten())
        self.layers.append(Linear(16 * 5 * 5, 120))
        self.layers.append(ReLU())
        self.layers.append(Linear(120, 84))
        self.layers.append(ReLU())
        self.layers.append(Linear(84, 10))
