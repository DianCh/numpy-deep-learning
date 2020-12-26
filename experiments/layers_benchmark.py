"""Script for Speed Benchmark experiment."""
from time import time

import numpy as np

from ndl.layers import (
    Conv2D,
    Conv2DThreeFold,
    Conv2DFourFold,
    Pool2D,
    Pool2DThreeFold,
    Pool2DFourFold,
)


def time_forward(layer, feature, num_iter, disp_name):
    """Time the forward pass of a layer."""
    tic = time()
    for _ in range(num_iter):
        _ = layer.forward(feature)
    print(f"{disp_name} used {time() - tic:.3f} seconds")


def conv2d_benchmark(feature, kwargs, num_iter):
    """Benchmark suites for Conv2D variants."""
    conv2d = Conv2D(**kwargs)
    time_forward(conv2d, feature, num_iter, "Conv2D")


def pool2d_benchmark():
    """Benchmark suites for Pool2D variants."""
    pass


if __name__ == "__main__":
    # hyper-params for the conv2d layer
    conv2d_kwargs = {
        "in_channels": 16,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
    }
    feature = np.random.randn(8, 64, 64, 16)
    conv2d_benchmark(feature, conv2d_kwargs, 2)
