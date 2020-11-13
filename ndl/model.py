import numpy as np


from layer import Conv2D, Linear, ReLU, Pool2D, Squeeze2D


class MultiLayerPerceptron:
    def __init__(self, in_features, out_features, hidden):
        self.channels = [in_features, *hidden]

        self.layers = []
        for l in range(len(self.channels) - 1):
            linear = Linear(self.channels[l], self.channels[l + 1], bias=True)
            relu = ReLU()

            self.layers.append(linear)
            self.layers.append(relu)

        linear = Linear(self.channels[-1], out_features, bias=True)
        self.layers.append(linear)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
