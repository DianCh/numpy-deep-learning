"""Implements Average/Max Pooling layer in 2D."""
import numpy as np

from ndl.utils import (
    get_tuple,
    get_output_shape,
    const_pad_tensor,
    unpad_tensor,
)


class Pool2D:
    """Standard 2D Avg/Max Pooling layer."""

    def __init__(self, kernel_size, stride, padding=(0, 0), mode="max"):
        assert mode in {
            "max",
            "avg",
        }, "Invalid mode for Pool2D. Only 'max' and 'avg' are supported."

        self.kernel_size = get_tuple(kernel_size)
        self.stride = get_tuple(stride)
        self.padding = get_tuple(padding)
        self.mode = mode

        self.cache_input = None
        self.cache_dim = None

    def clear_cache(self):
        """Clear forward pass cache."""
        self.cache_input = None
        self.cache_dim = None

    def forward(self, x):
        """Forward computation of pooling."""
        # determine input dimensions
        m, H_prev, W_prev, C = x.shape

        # determine output dimensions
        H, W = get_output_shape(
            self.kernel_size, self.stride, self.padding, H_prev, W_prev
        )

        # pad input tensor
        x_pad = const_pad_tensor(x, self.padding)

        # save cache for back-propagation
        self.cache_input = x_pad
        self.cache_dim = x.shape

        out = self._forward_compute(x_pad, m, H, W, C)

        return out

    def _forward_compute(self, x_pad, m, H, W, C):
        """2-fold for loop implementation."""
        # initialize container for output
        out = np.zeros((m, H, W, C))

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

                # (m, k, k, C)
                x_slice = x_pad[:, h_start:h_end, w_start:w_end, :]
                if self.mode == "max":
                    out[:, h, w, :] = np.max(x_slice, axis=(1, 2))
                if self.mode == "avg":
                    out[:, h, w, :] = np.mean(x_slice, axis=(1, 2))

        return out

    def backward(self, dY):
        """Backward computation of gradients."""
        assert (
            self.cache_input is not None
        ), "Cannot backprop without forward first."

        # retrieve input & dimensions
        x_pad = self.cache_input
        m, H_prev, W_prev, C = self.cache_dim

        # clear cache
        self.clear_cache()

        # initialize & pad containers for gradients w.r.t input
        dX = np.zeros((m, H_prev, W_prev, C))
        dX_pad = const_pad_tensor(dX, self.padding)

        dX = self._backward_compute(x_pad, dX_pad, dY, H_prev, W_prev)

        return dX

    def _backward_compute(self, x_pad, dX_pad, dY, H_prev, W_prev):
        """2-fold for loop implementation."""
        m, H, W, C = dY.shape

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
                    avg_volume = np.ones((m, k_H, k_W, C)) / (k_H * k_W)
                    dX_pad[:, h_start:h_end, w_start:w_end, :] += (
                        dY_ * avg_volume
                    )

        # slice the gradient tensor to original size
        dX = unpad_tensor(dX_pad, self.padding, (H_prev, W_prev))

        return dX


class Pool2DThreeFold(Pool2D):
    """Standard 2D Avg/Max Pooling layer with 3-fold for loop implementation."""

    def _forward_compute(self, x_pad, m, H, W, C):
        """3-fold for loop implementation."""
        # initialize container for output
        out = np.zeros((m, H, W, C))

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

                for c in range(C):
                    # (m, k, k)
                    x_slice = x_pad[:, h_start:h_end, w_start:w_end, c]
                    if self.mode == "max":
                        out[:, h, w, c] = np.max(x_slice, axis=(1, 2))
                    if self.mode == "avg":
                        out[:, h, w, c] = np.mean(x_slice, axis=(1, 2))

        return out

    def _backward_compute(self, x_pad, dX_pad, dY, H_prev, W_prev):
        """2-fold for loop implementation."""
        m, H, W, C = dY.shape

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

                for c in range(C):
                    # (m, k, k)
                    x_slice = x_pad[:, h_start:h_end, w_start:w_end, c]
                    # (m, 1, 1)
                    dY_ = np.expand_dims(dY[:, h, w, c], axis=(1, 2))

                    if self.mode == "max":
                        mask = x_slice == np.max(
                            x_slice, axis=(1, 2), keepdims=True
                        )
                        dX_pad[:, h_start:h_end, w_start:w_end, c] += dY_ * mask

                    elif self.mode == "avg":
                        avg_volume = np.ones((m, k_H, k_W)) / (k_H * k_W)
                        dX_pad[:, h_start:h_end, w_start:w_end, c] += (
                            dY_ * avg_volume
                        )

        # slice the gradient tensor to original size
        dX = unpad_tensor(dX_pad, self.padding, (H_prev, W_prev))

        return dX


class Pool2DFourFold(Pool2D):
    """Standard 2D Avg/Max Pooling layer with 4-fold for loop implementation."""

    def _forward_compute(self, x_pad, m, H, W, C):
        """4-fold for loop implementation."""
        # initialize container for output
        out = np.zeros((m, H, W, C))

        stride_H, stride_W = self.stride
        k_H, k_W = self.kernel_size
        for i in range(m):
            for h in range(H):
                # slice boundaries in H direction
                h_start = h * stride_H
                h_end = h * stride_H + k_H
                for w in range(W):
                    # slice boundaries in W direction
                    w_start = w * stride_W
                    w_end = w * stride_W + k_W

                    for c in range(C):
                        # (k, k)
                        x_slice = x_pad[i, h_start:h_end, w_start:w_end, c]
                        if self.mode == "max":
                            out[i, h, w, c] = np.max(x_slice)
                        if self.mode == "avg":
                            out[i, h, w, c] = np.mean(x_slice)

        return out

    def _backward_compute(self, x_pad, dX_pad, dY, H_prev, W_prev):
        """2-fold for loop implementation."""
        m, H, W, C = dY.shape

        stride_H, stride_W = self.stride
        k_H, k_W = self.kernel_size
        for i in range(m):
            for h in range(H):
                # slice boundaries in H direction
                h_start = h * stride_H
                h_end = h * stride_H + k_H
                for w in range(W):
                    # slice boundaries in W directions
                    w_start = w * stride_W
                    w_end = w * stride_W + k_W

                    for c in range(C):
                        # (k, k)
                        x_slice = x_pad[i, h_start:h_end, w_start:w_end, c]

                        if self.mode == "max":
                            mask = x_slice == np.max(x_slice)
                            dX_pad[i, h_start:h_end, w_start:w_end, c] += (
                                dY[i, h, w, c] * mask
                            )

                        elif self.mode == "avg":
                            avg_volume = np.ones((k_H, k_W)) / (k_H * k_W)
                            dX_pad[i, h_start:h_end, w_start:w_end, c] += (
                                dY[i, h, w, c] * avg_volume
                            )

        # slice the gradient tensor to original size
        dX = unpad_tensor(dX_pad, self.padding, (H_prev, W_prev))

        return dX
