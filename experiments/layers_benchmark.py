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


def verify_forward(targets, feature):
    """Verify different implementations of forward are correct."""
    assert len(targets) > 1, f"Need at least 2 variants to compare."
    outputs = [layer.forward(feature) for layer, _ in targets]

    for i in range(len(targets) - 1):
        assert np.allclose(
            outputs[i], outputs[i + 1], atol=1e-7
        ), f"Results of {targets[i][1]} and {targets[i + 1][1]} are different."

    print(f"All variants are consistent.")


def time_forward(target, feature, num_iter):
    """Time the forward pass of a layer."""
    layer, disp_name = target
    tic = time()
    for _ in range(num_iter):
        _ = layer.forward(feature)
    print(
        f"{disp_name} used {time() - tic:.3f} seconds for",
        f"{num_iter} forward passes",
    )


def conv2d_benchmark(feature, kwargs, num_iter):
    """Benchmark suites for Conv2D variants."""
    # create Conv2D variants
    conv2d = Conv2D(**kwargs)
    conv2d_im2col = Conv2D(**kwargs, use_im2col=True)
    conv2d_threefold = Conv2DThreeFold(**kwargs)
    conv2d_fourfold = Conv2DFourFold(**kwargs)

    targets = [
        (conv2d, "Conv2D"),
        (conv2d_im2col, "Conv2D with im2col"),
        (conv2d_threefold, "Conv2DThreeFold"),
        (conv2d_fourfold, "Conv2DFourFold"),
    ]

    # set unified weights & biases
    for i in range(len(targets) - 1):
        targets[i + 1][0].W[:] = targets[i][0].W.copy()
        targets[i + 1][0].b[:] = targets[i][0].b.copy()

    verify_forward(targets, feature)

    for target in targets:
        time_forward(target, feature, num_iter)


def pool2d_benchmark():
    """Benchmark suites for Pool2D variants."""
    pass


if __name__ == "__main__":
    # hyper-params for the conv2d layer
    conv2d_kwargs = {
        "in_channels": 16,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
    }
    feature = np.random.randn(8, 64, 64, 16)
    conv2d_benchmark(feature, conv2d_kwargs, 5)
