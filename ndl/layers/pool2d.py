"""Implements Average/Max Pooling layer in 2D."""
import numpy as np

from ndl.utils import (
    get_tuple,
    get_output_shape,
    const_pad_tensor,
    unpad_tensor,
)


class Pool2D:
    """Standard 2D Average/Max Pooling layer."""

    def __init__(self, kernel_size, stride, padding=(0, 0), mode="max"):
        assert mode in {
            "max",
            "avg",
        }, "Invalid mode for Pool2D. Only 'max' and 'avg' are supported."

        self.kernel_size = get_tuple(kernel_size)
        self.stride = get_tuple(stride)
        self.padding = get_tuple(padding)
        self.mode = mode

        self.cache = None

    def forward(self, x):
        """Forward computation of pooling."""
        # determine input dimensions
        m, H_prev, W_prev, C_prev = x.shape

        # determine output dimensions
        H, W = get_output_shape(
            self.kernel_size, self.stride, self.padding, H_prev, W_prev
        )

        # initialize container for output
        out = np.zeros((m, H, W, C_prev))

        # pad input tensor
        x_pad = const_pad_tensor(x, self.padding)

        stride_H, stride_W = self.stride
        k_H, k_W = self.kernel_size
        for h in range(H):
            # slice boundaries in H direction
            h_start = h * stride_H
            h_end = h * stride_H + k_H
            for w in range(W):
                # slice boundaries in W direction
                w_start = w * stride_W
                w_end = w * stride_W + k_W

                x_slice = x_pad[:, h_start:h_end, w_start:w_end, :]
                if self.mode == "max":
                    out[:, h, w, :] = np.max(x_slice, axis=(1, 2))
                if self.mode == "avg":
                    out[:, h, w, :] = np.mean(x_slice, axis=(1, 2))

        # save cache for back-propagation
        self.cache = x

        return out

    def backward(self, dY):
        """Backward computation of gradients."""
        assert self.cache is not None, "Cannot backprop without forward first."
        x = self.cache

        # retrieve input & output dimensions
        m, H_prev, W_prev, C_prev = x.shape
        _, H, W, _ = dY.shape

        # initialize & pad containers for gradients w.r.t input
        dX = np.zeros((m, H_prev, W_prev, C_prev))
        dX_pad = const_pad_tensor(dX, self.padding)
        x_pad = const_pad_tensor(x, self.padding)

        stride_H, stride_W = self.stride
        k_H, k_W = self.kernel_size
        for h in range(H):
            # slice boundaries in H direction
            h_start = h * stride_H
            h_end = h * stride_H + k_H
            for w in range(W):
                # slice boundaries in W directions
                w_start = w * stride_W
                w_end = w * stride_W + k_W

                # (m, k, k, C_prev)
                x_slice = x_pad[:, h_start:h_end, w_start:w_end, :]
                # (m, 1, 1, C_prev)
                dY_ = np.expand_dims(dY[:, h, w, :], axis=(1, 2))

                if self.mode == "max":
                    mask = x_slice == np.max(
                        x_slice, axis=(1, 2), keepdims=True
                    )
                    dX_pad[:, h_start:h_end, w_start:w_end, :] += dY_ * mask

                elif self.mode == "avg":
                    avg_volume = np.ones((m, k_H, k_W, C_prev)) / (k_H * k_W)
                    dX_pad[:, h_start:h_end, w_start:w_end, :] += (
                        dY_ * avg_volume
                    )

        # slice the gradient tensor to original size
        dX = unpad_tensor(dX_pad, self.padding, (H_prev, W_prev))

        # clear cache
        self.cache = None

        return dX


class Pool2DThreeFold(Pool2D):
    pass


class Pool2DFourFold(Pool2D):
    pass