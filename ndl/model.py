import numpy as np


from ndl.layer import Conv2D, Linear, ReLU, Pool2D, Squeeze2D, Flatten


class Sequential:
    def __init__(self):
        self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


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

        self.layers.append(Conv2D(3, 32, 3, 1, 1))
        self.layers.append(Pool2D(2, 2, 0, "max"))
        self.layers.append(ReLU())

        self.layers.append(Conv2D(32, 64, 3, 1, 1))
        self.layers.append(Pool2D(2, 2, 0, "max"))
        self.layers.append(ReLU())

        self.layers.append(Conv2D(64, 64, 3, 1, 1))
        self.layers.append(ReLU())

        self.layers.append(Flatten())

        self.layers.append(Linear(1024, 64, True))
        self.layers.append(ReLU())
        self.layers.append(Linear(64, 10, True))
