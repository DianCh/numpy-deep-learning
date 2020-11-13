import numpy as np


def get_tuple(number):
    if isinstance(number, int):
        return (number, number)
    return tuple(number)


def get_output_shape(kernel_size, stride, padding, H_prev, W_prev):
    k_H, k_W = kernel_size
    stride_H, stride_W = stride
    pad_H, pad_W = padding

    H = int((H_prev - k_H + 2 * pad_H) / stride_H) + 1
    W = int((W_prev - k_W + 2 * pad_W) / stride_W) + 1

    return H, W


def const_pad_tensor(x, padding, value=0):
    pad_H, pad_W = get_tuple(padding)
    x_pad = np.pad(
        x,
        ((0, 0), (pad_H, pad_H), (pad_W, pad_W), (0, 0)),
        mode="constant",
        constant_values=(value, value),
    )

    return x_pad


def unpad_tensor(x, padding, shape):
    pad_H, pad_W = padding
    H, W = shape

    h_start, h_end = (pad_H, -pad_H) if pad_H > 0 else (0, H)
    w_start, w_end = (pad_W, -pad_W) if pad_W > 0 else (0, W)

    return x[:, h_start:h_end, w_start:w_end, :]


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = get_tuple(kernel_size)
        self.stride = get_tuple(stride)
        self.padding = get_tuple(padding)

        self.W = np.random.randn(*self.kernel_size, in_channels, out_channels)
        self.b = np.zeros((out_channels,))

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
                out[:, h, w, :] = np.sum(weights * x_slice, axis=(1, 2, 3)) + self.b

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

                # (m, k, k, C_prev, C)
                weights = np.repeat(np.expand_dims(self.W, 0), repeats=m, axis=0)
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


class Pool2D:
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
        # determine input dimensions
        m, H_prev, W_prev, C_prev = x.shape

        # determine output dimensions
        H, W = get_output_shape(
            self.kernel_size, self.padding, self.stride, H_prev, W_prev
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
                out[:, h, w, :] = np.max(x_slice, axis=(1, 2))

        # save cache for back-propagation
        self.cache = x

        return out

    def backward(self, dY):
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
                    mask = x_slice == np.max(x_slice, axis=(1, 2), keepdims=True)
                    dX_pad[:, h_start:h_end, w_start:w_end, :] += dY_ * mask

                elif self.mode == "avg":
                    avg_volume = np.ones((m, k_H, k_W, C_prev)) / (k_H * k_W)
                    dX_pad[:, h_start:h_end, w_start:w_end, :] += dY_ * avg_volume

        # slice the gradient tensor to original size
        dX = unpad_tensor(dX_pad, self.padding, (H_prev, W_prev))

        # clear cache
        self.cache = None

        return dX


class Linear:
    def __init__(self, in_features, out_features, bias):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.W = np.random.randn(in_features, out_features)
        self.b = np.zeros((out_features,))

        self.dW = np.zeros((in_features, out_features))
        self.db = np.zeros((out_features,))

        self.cache = None

    def clear_gradients(self):
        self.dW *= 0.0
        self.db *= 0.0

    def forward(self, x):
        # support 1-dimensional sample feature for now
        assert x.ndim == 2, "Only 1-dimensional feature is supported for now."

        # (m, C_prev) x (C_prev, C) + (C, )
        out = np.dot(x, self.W) + self.b

        # save cache for back-propagation
        self.cache = x

        return out

    def backward(self, dY):
        # clear existing gradients
        self.clear_gradients()

        assert self.cache is not None, "Cannot backprop without forward first."
        x = self.cache

        self.dW = np.dot(x.T, dY)
        if self.bias:
            self.db = np.sum(dY, axis=0)

        dX = np.dot(dY, self.W.T)

        # clear cache
        self.cache = None
        return dX


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        # cap negative values to zero
        mask = x >= 0.0
        out = x * mask

        # save cache for back-propagation
        self.cache = mask

        return out

    def backward(self, dY):
        assert self.cache is not None, "Cannot backprop without forward first."
        mask = self.cache

        # shut down gradients at negative positions
        dX = dY * mask

        # clear cache
        self.cache = None

        return dX


class Sigmoid:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))

        # save cache for back-propagation
        self.cache = out

        return out

    def backward(self, dY):
        assert self.cache is not None, "Cannot backprop without forward first."
        y = self.cache

        dX = dY * y * (1 - y)

        # clear cache
        self.cache = None

        return dX


class Squeeze2D:
    def forward(self, x):
        return x.squeeze()

    def backward(self, dY):
        return np.expand_dims(dY, axis=(1, 2))


class Flatten:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        shape = x.shape

        # save cache for back-propagation
        self.cache = shape

        return x.reshape((shape[0], -1))

    def backward(self, dY):
        assert self.cache is not None, "Cannot backprop without forward first."
        shape = self.cache

        # clear cache
        self.cache = None

        return dY.reshape(shape)
