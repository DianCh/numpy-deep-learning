import numpy as np


def get_tuple(number):
    if isinstance(number, int):
        return (number, number)
    return tuple(number)


def zero_pad_tensor(x, padding, value=0):
    p_H, p_W = get_tuple(padding)
    x_pad = np.pad(
        x,
        ((0, 0), (p_H, p_H), (p_W, p_W), (0, 0)),
        mode="constant",
        constant_values=(value, value),
    )

    return x_pad


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = get_tuple(kernel_size)
        self.stride = get_tuple(stride)
        self.padding = get_tuple(padding)

        self.weights = np.random.randn(*self.kernel_size, in_channels, out_channels)
        self.biases = np.zeros((out_channels,))

        self.dW = np.zeros((*self.kernel_size, in_channels, out_channels))
        self.db = np.zeros((out_channels,))

        self.cache = None

    def clear_gradients(self):
        self.dW *= 0.0
        self.db *= 0.0

    def forward(self, x):
        # determine input dimensions
        m, H_prev, W_prev, _ = x.shape

        # determin output dimensions
        k_H, k_W = self.kernel_size
        pad_H, pad_W = self.padding
        stride_H, stride_W = self.stride
        H = int((H_prev - k_H + 2 * pad_H) / stride_H) + 1
        W = int((W_prev - k_W + 2 * pad_W) / stride_W) + 1

        # initialize container for output
        out = np.zeros((m, H, W, self.out_channels))

        x_pad = zero_pad_tensor(x, self.padding)
        # (1, k, k, in_channels, out_channels)
        weights = np.expand_dims(self.weights, axis=0)

        for h in range(H):
            # slice boundaries in H direction
            h_start = h * stride_H
            h_end = h * stride_H + k_H
            for w in range(W):
                # slice boundaries in W direction
                w_start = w * stride_W
                w_end = w * stride_W + k_W

                # (m, k, k, in_channels, 1)
                x_slice = x_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis]
                # (m, out_channels)
                out[:, h, w, :] = (
                    np.sum(weights * x_slice, axis=(1, 2, 3)) + self.biases
                )

        # save cache for back-propagation
        self.cache = x

        return out

    def backward(self, dY):
        # clear existing
        self.clear_gradients()

        assert self.cache is not None, "Cannot backprop without forward first."
        x = self.cache

        # retrieve input & output dimensions
        m, H_prev, W_prev, in_channels = x.shape
        _, H, W, _ = dY.shape

        # initialize & pad containers for gradients w.r.t input
        dX = np.zeros((m, H_prev, W_prev, in_channels))
        dX_pad = zero_pad_tensor(dX, self.padding)
        x_pad = zero_pad_tensor(x, self.padding)

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

                # (m, k, k, in_channels, out_channels)
                weights = np.repeat(np.expand_dims(self.weights, 0), repeats=m, axis=0)
                # (m, 1, 1, 1, out_channels)
                dY_ = np.expand_dims(dY[:, h, w, :], axis=(1, 2, 3))
                # (m, k, k, in_channels)
                dX_pad[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    weights * dY_, axis=4
                )

                # (m, k, k, in_channels, 1)
                x_slice = x_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis]
                # (k, k, in_channels, out_channels)
                self.dW += np.sum(x_slice * dY_, axis=0)
                # (out_channels, )
                self.db += np.sum(dY_, axis=(0, 1, 2, 3))

        # slice the gradient tensor at original size
        pad_H, pad_W = self.padding
        dX = dX_pad[:, pad_H:-pad_H, pad_W:-pad_W, :]

        # clear cache
        self.cache = None

        return dX
