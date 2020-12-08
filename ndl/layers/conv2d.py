"""Implements 2D Convolution layer."""
import numpy as np

from ndl.layers import Base
from ndl.utils import (
    get_tuple,
    get_output_shape,
    const_pad_tensor,
    unpad_tensor,
)


class Conv2D(Base):
    """Standard 2D Convolution layer."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = get_tuple(kernel_size)
        self.stride = get_tuple(stride)
        self.padding = get_tuple(padding)

        self.W = (
            np.random.randn(*self.kernel_size, in_channels, out_channels)
            * 0.00001
        )
        self.b = np.zeros((out_channels,))

        self.dW = np.zeros((*self.kernel_size, in_channels, out_channels))
        self.db = np.zeros((out_channels,))

        self.cache = None

        self.kaiming_uniform_init_weights()
        self.kaiming_uniform_init_biases()

    def forward(self, x):
        """Forward computation of convolution."""
        # determine input dimensions
        m, H_prev, W_prev, _ = x.shape

        # determin output dimensions
        H, W = get_output_shape(
            self.kernel_size, self.stride, self.padding, H_prev, W_prev
        )

        # initialize container for output
        out = np.zeros((m, H, W, self.out_channels))

        # pad input tensor
        x_pad = const_pad_tensor(x, self.padding)

        # (1, k, k, C_prev, C)
        weights = np.expand_dims(self.W, axis=0)
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

                # (m, k, k, C_prev, 1)
                x_slice = x_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis]
                # (m, C)
                out[:, h, w, :] = (
                    np.sum(weights * x_slice, axis=(1, 2, 3)) + self.b
                )

        # save cache for back-propagation
        self.cache = x

        return out

    def backward(self, dY):
        """Backward computation of gradients."""
        # clear existing gradients
        self.clear_gradients()

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

                # (m, k, k, C_prev, C)
                weights = np.repeat(
                    np.expand_dims(self.W, 0), repeats=m, axis=0
                )
                # (m, 1, 1, 1, C)
                dY_ = np.expand_dims(dY[:, h, w, :], axis=(1, 2, 3))
                # (m, k, k, C_prev)
                dX_pad[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    weights * dY_, axis=4
                )

                # (m, k, k, C_prev, 1)
                x_slice = x_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis]
                # (k, k, C_prev, C)
                self.dW += np.sum(x_slice * dY_, axis=0)
                # (C, )
                self.db += np.sum(dY_, axis=(0, 1, 2, 3))

        # slice the gradient tensor to original size
        dX = unpad_tensor(dX_pad, self.padding, (H_prev, W_prev))

        # clear cache
        self.cache = None

        return dX


class Conv2DThreeFold(Conv2D):
    def forward(self, x):
        # determine input dimensions
        m, H_prev, W_prev, _ = x.shape

        # determin output dimensions
        H, W = get_output_shape(
            self.kernel_size, self.stride, self.padding, H_prev, W_prev
        )

        # initialize container for output
        out = np.zeros((m, H, W, self.out_channels))

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

                # (m, k, k, C_prev)
                x_slice = x_pad[:, h_start:h_end, w_start:w_end, :]
                # loop over output channels
                for c in range(self.out_channels):
                    out[:, h, w, c] = (
                        np.sum(self.W[:, :, :, c] * x_slice, axis=(1, 2, 3))
                        + self.b[c]
                    )  # (k, k, C_prev) x (m, k, k, C_prev)

        # save cache for back-propagation
        self.cache = x

        return out

    def backward(self, dY):
        # clear existing gradients
        self.clear_gradients()

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
                # loop over output channels
                for c in range(self.out_channels):
                    # (m, k, k, C_prev)
                    weights = np.repeat(
                        self.W[np.newaxis, ..., c], repeats=m, axis=0
                    )
                    # (m, 1, 1, 1)
                    dY_ = np.expand_dims(dY[:, h, w, c], axis=(1, 2, 3))
                    # (m, k, k, C_prev)
                    dX_pad[:, h_start:h_end, w_start:w_end, :] += weights * dY_

                    # (k, k, C_prev)
                    self.dW[..., c] += np.sum(x_slice * dY_, axis=0)
                    # (1, )
                    self.db[c] += np.sum(dY_)

        # slice the gradient tensor to original size
        dX = unpad_tensor(dX_pad, self.padding, (H_prev, W_prev))

        # clear cache
        self.cache = None

        return dX


class Conv2DFourFold(Conv2D):
    def forward(self, x):
        pass

    def backward(self, dY):
        pass

