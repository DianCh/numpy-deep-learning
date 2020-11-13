import numpy as np


from layer import Conv2D, Linear, ReLU, Pool2D, Squeeze2D


class MultiLayerPerceptron:
    def __init__(self, in_features, out_features, hidden):
        self.channels = [in_features, *hidden, out_features]

        self.layers = []
        for l in range(len(self.channels) - 1):
            linear = Linear(self.channels[l], self.channels[l + 1], bias=True)
            relu = ReLU()

            self.layers.append(linear)
            self.layers.append(relu)

    def forward(self, x):
        print("hahahaha", x)
        for layer in self.layers:
            # print("hahahaha", x)
            x = layer.forward(x)
        return x
