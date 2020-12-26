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
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = get_tuple(kernel_size)
        self.stride = get_tuple(stride)
        self.padding = get_tuple(padding)

        self.W = (
            np.random.randn(*self.kernel_size, in_channels, out_channels)
            * 0.001
        )
        self.b = np.zeros((out_channels,))

        self.dW = np.zeros((*self.kernel_size, in_channels, out_channels))
        self.db = np.zeros((out_channels,))

        self.cache_input = None
        self.cache_dim = None

        self.kaiming_uniform_init_weights()
        self.kaiming_uniform_init_biases()

    def clear_cache(self):
        """Clear forward pass cache."""
        self.cache_input = None
        self.cache_dim = None

    def forward(self, x):
        """Forward computation of convolution."""
        # save cache for back-propagation
        self.cache_dim = x.shape

        # determin output dimensions
        m, H_prev, W_prev, _ = x.shape
        H, W = get_output_shape(
            self.kernel_size, self.stride, self.padding, H_prev, W_prev
        )

        # pad input tensor
        x_pad = const_pad_tensor(x, self.padding)

        out = self._foward_compute(x_pad, m, H, W)

        return out

    def _foward_compute(self, x_pad, m, H, W):
        """im2col implementation."""
        # flatten weights into 2D matrix
        # (k, k, C_prev, C) -> (k x k x C_prev, C) -> (1, k x k x C_prev, C)
        weights = self.W.reshape((-1, self.out_channels))[np.newaxis, ...]

        # im2col matrix
        stride_H, stride_W = self.stride
        k_H, k_W = self.kernel_size
        col_matrix = np.zeros((m, H * W, k_H * k_W * self.in_channels))
        for h in range(H):
            # slice boundaries in H direction
            h_start = h * stride_H
            h_end = h * stride_H + k_H
            for w in range(W):
                # slice boundaries in W direction
                w_start = w * stride_W
                w_end = w * stride_W + k_W

                idx = h * H + w
                # (m, k, k, C_prev) -> (m, k x k x C_prev)
                row = x_pad[:, h_start:h_end, w_start:w_end, :].reshape((m, -1))
                col_matrix[:, idx, :] = row

        # with im2col implementation, use col_matrix
        self.cache_input = col_matrix

        # (m, H x W, C) -> (m, H, W, C)
        out = np.matmul(col_matrix, weights).reshape(
            (m, H, W, self.out_channels)
        )
        out += self.b

        return out

    def backward(self, dY):
        """Backward computation of gradients."""
        # clear existing gradients
        self.clear_gradients()

        assert (
            self.cache_input is not None
        ), "Cannot backprop without forward first."

        # retrieve input & dimensions
        cache_input = self.cache_input
        m, H_prev, W_prev, C_prev = self.cache_dim

        # clear cache
        self.clear_cache()

        # initialize & pad containers for gradients w.r.t input
        dX = np.zeros((m, H_prev, W_prev, C_prev))
        dX_pad = const_pad_tensor(dX, self.padding)

        dX = self._backward_compute(cache_input, dX_pad, dY, H_prev, W_prev)

        return dX

    def _backward_compute(self, col_matrix, dX_pad, dY, H_prev, W_prev):
        """im2col implementation."""
        m, H, W, _ = dY.shape

        # (m, H, W, C) -> (m, H x W, C)
        dY = dY.reshape((m, -1, self.out_channels))
        # (k, k, C_prev, C) -> (k x k x C_prev, C) -> (1, k x k x C_prev, C)
        weights = self.W.reshape((-1, self.out_channels))[np.newaxis, ...]

        # gradients of activation
        stride_H, stride_W = self.stride
        k_H, k_W = self.kernel_size
        dcol_matrix = np.matmul(dY, weights.transpose(0, 2, 1))
        for h in range(H):
            # slice boundaries in H direction
            h_start = h * stride_H
            h_end = h * stride_H + k_H
            for w in range(W):
                # slice boundaries in W direction
                w_start = w * stride_W
                w_end = w * stride_W + k_W

                idx = h * H + w
                drow = dcol_matrix[:, idx, :].reshape((m, k_H, k_W, -1))
                dX_pad[:, h_start:h_end, w_start:w_end, :] = drow

        # slice the gradient tensor to original size
        dX = unpad_tensor(dX_pad, self.padding, (H_prev, W_prev))

        # gradients of weights & biases
        dW = np.sum(np.matmul(col_matrix.transpose(0, 2, 1), dY), axis=0)
        self.dW[:] = dW.reshape(
            (*self.kernel_size, self.in_channels, self.out_channels)
        )
        self.db[:] = np.sum(dY, axis=(0, 1))

        return dX


class Conv2DTwoFold(Conv2D):
    """Standard 2D Convolution layer with 2-fold for loop implementation."""

    def _foward_compute(self, x_pad, m, H, W):
        """2-fold for loop implementation."""
        # with for loop implementgation, use x_pad
        self.cache_input = x_pad

        # initialize container for output
        out = np.zeros((m, H, W, self.out_channels))

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

        return out

    def _backward_compute(self, x_pad, dX_pad, dY, H_prev, W_prev):
        """2-fold for loop implementation."""
        m, H, W, _ = dY.shape

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

        return dX


class Conv2DThreeFold(Conv2D):
    """Standard 2D Convolution layer with 3-folder for loop implementation."""

    def _foward_compute(self, x_pad, m, H, W):
        """3-fold for loop implementation."""
        # initialize container for output
        out = np.zeros((m, H, W, self.out_channels))

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

        return out

    def _backward_compute(self, x_pad, dX_pad, dY, H_prev, W_prev):
        """3-fold for loop implementation."""
        m, H, W, _ = dY.shape

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

        return dX


class Conv2DFourFold(Conv2D):
    """Standard 2D Convolution layer with 4-fold for loop implementation."""

    def _foward_compute(self, x_pad, m, H, W):
        """4-fold for loop implementation."""
        # initialize container for output
        out = np.zeros((m, H, W, self.out_channels))

        stride_H, stride_W = self.stride
        k_H, k_W = self.kernel_size
        # loop over samples
        for i in range(m):
            for h in range(H):
                # slice boundaries in H direction
                h_start = h * stride_H
                h_end = h * stride_H + k_H
                for w in range(W):
                    # slice boundaries in W direction
                    w_start = w * stride_W
                    w_end = w * stride_W + k_W

                    # (k, k, C_prev)
                    x_slice = x_pad[i, h_start:h_end, w_start:w_end, :]
                    # loop over output channels
                    for c in range(self.out_channels):
                        out[i, h, w, c] = (
                            np.sum(self.W[:, :, :, c] * x_slice) + self.b[c]
                        )  # (k, k, C_prev) x (k, k, C_prev)

        return out

    def _backward_compute(self, x_pad, dX_pad, dY, H_prev, W_prev):
        """4-fold for loop implementation."""
        m, H, W, _ = dY.shape

        stride_H, stride_W = self.stride
        k_H, k_W = self.kernel_size
        # loop over samples
        for i in range(m):
            for h in range(H):
                # slice boundaries in H direction
                h_start = h * stride_H
                h_end = h * stride_H + k_H
                for w in range(W):
                    # slice boundaries in W directions
                    w_start = w * stride_W
                    w_end = w * stride_W + k_W

                    # (k, k, C_prev)
                    x_slice = x_pad[i, h_start:h_end, w_start:w_end, :]
                    # loop over output channels
                    for c in range(self.out_channels):
                        # (k, k, C_prev)
                        weights = self.W[..., c]

                        # (k, k, C_prev)
                        dX_pad[i, h_start:h_end, w_start:w_end, :] += (
                            weights * dY[i, h, w, c]
                        )

                        # (k, k, C_prev)
                        self.dW[..., c] += x_slice * dY[i, h, w, c]
                        # (1, )
                        self.db[c] += dY[i, h, w, c]

            # slice the gradient tensor to original size
            dX = unpad_tensor(dX_pad, self.padding, (H_prev, W_prev))

        return dX
