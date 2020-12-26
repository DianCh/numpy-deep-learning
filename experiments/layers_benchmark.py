"""Script for Speed Benchmark experiment."""
from time import time

import numpy as np

from ndl.layers import (
    Conv2D,
    Conv2DTwoFold,
    Conv2DThreeFold,
    Conv2DFourFold,
    Pool2D,
    Pool2DThreeFold,
    Pool2DFourFold,
)


def verify_results(targets, sample, forward=True):
    """Verify different implementations of forward/backward are correct."""
    assert len(targets) > 1, f"Need at least 2 variants to compare."
    outputs = (
        [layer.forward(sample) for layer, _ in targets]
        if forward
        else [layer.backward(sample) for layer, _ in targets]
    )

    for i in range(len(targets) - 1):
        assert np.allclose(
            outputs[i], outputs[i + 1], atol=1e-7
        ), f"Results of {targets[i][1]} and {targets[i + 1][1]} are different."

    test_name = "forward" if forward else "backward"
    print(f"All variants' {test_name} are consistent.")


def time_compute(target, feature, gradient, num_iter):
    """Time the forward & backward pass of a layer."""
    layer, disp_name = target
    forward_time = 0
    backward_time = 0
    for _ in range(num_iter):
        # forward
        tic = time()
        _ = layer.forward(feature)
        toc = time()

        # backward
        _ = layer.backward(gradient)
        tac = time()
        forward_time += toc - tic
        backward_time += tac - toc

    print(
        f"{disp_name} used {forward_time:.3f} seconds for",
        f"{num_iter} forward passes,",
        f"{backward_time:.3f} seconds for {num_iter} backward passes.",
    )


def conv2d_benchmark(feature, gradient, kwargs, num_iter):
    """Benchmark suites for Conv2D variants."""
    # create Conv2D variants
    conv2d = Conv2D(**kwargs)
    conv2d_twofold = Conv2DTwoFold(**kwargs)
    conv2d_threefold = Conv2DThreeFold(**kwargs)
    conv2d_fourfold = Conv2DFourFold(**kwargs)

    targets = [
        (conv2d, "Conv2D with im2col"),
        (conv2d_twofold, "Conv2DTwoFold"),
        (conv2d_threefold, "Conv2DThreeFold"),
        (conv2d_fourfold, "Conv2DFourFold"),
    ]

    # set unified weights & biases
    for i in range(len(targets) - 1):
        targets[i + 1][0].W[:] = targets[i][0].W.copy()
        targets[i + 1][0].b[:] = targets[i][0].b.copy()

    # verify implementation
    verify_results(targets, feature)
    verify_results(targets, gradient, forward=False)

    # timing
    for target in targets:
        time_compute(target, feature, gradient, num_iter)


def pool2d_benchmark(feature, gradient, kwargs, num_iter):
    """Benchmark suites for Pool2D variants."""
    # create Pool2D variants
    pool2d = Pool2D(**kwargs)
    pool2d_threefold = Pool2DThreeFold(**kwargs)
    pool2d_fourfold = Pool2DFourFold(**kwargs)

    targets = [
        (pool2d, "Pool2D"),
        (pool2d_threefold, "Pool2DThreeFold"),
        (pool2d_fourfold, "Pool2DFourFold"),
    ]

    # verify implementation
    verify_results(targets, feature)
    verify_results(targets, gradient, forward=False)

    # timing
    for target in targets:
        time_compute(target, feature, gradient, num_iter)


if __name__ == "__main__":
    # hyper-params & test tensors for the conv2d layers
    conv2d_kwargs = {
        "in_channels": 16,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
    }
    feature = np.random.randn(8, 64, 64, 16)
    gradient = np.random.randn(8, 32, 32, 32)

    print("Benchmarking Conv2D...")
    conv2d_benchmark(feature, gradient, conv2d_kwargs, 3)

    # hyper-params & test tensors for the pool2d layers
    pool2d_kwargs = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
    }
    feature = np.random.randn(8, 64, 64, 16)
    gradient = np.random.randn(8, 32, 32, 16)

    print("Benchmarking Max Pool2D...")
    pool2d_benchmark(feature, gradient, pool2d_kwargs, 3)

    print("Benchmarking Avg Pool2D...")
    pool2d_kwargs["mode"] = "avg"
    pool2d_benchmark(feature, gradient, pool2d_kwargs, 3)
