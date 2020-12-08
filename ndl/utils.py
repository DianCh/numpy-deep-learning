import numpy as np


def get_tuple(number, repeat=2):
    """Make a tuple from input number."""
    if isinstance(number, int):
        return (number,) * repeat
    return tuple(number)


def get_output_shape(kernel_size, stride, padding, H_prev, W_prev):
    """Get the output shape given kernal size, stride, padding, input size."""
    k_H, k_W = kernel_size
    stride_H, stride_W = stride
    pad_H, pad_W = padding

    H = int((H_prev - k_H + 2 * pad_H) / stride_H) + 1
    W = int((W_prev - k_W + 2 * pad_W) / stride_W) + 1

    return H, W


def const_pad_tensor(x, padding, value=0):
    """Pad a tensor with padding size and constant value."""
    pad_H, pad_W = get_tuple(padding)
    x_pad = np.pad(
        x,
        ((0, 0), (pad_H, pad_H), (pad_W, pad_W), (0, 0)),
        mode="constant",
        constant_values=(value, value),
    )

    return x_pad


def unpad_tensor(x, padding, shape):
    """Strip away padded values around a tensor."""
    pad_H, pad_W = padding
    H, W = shape

    h_start, h_end = (pad_H, -pad_H) if pad_H > 0 else (0, H)
    w_start, w_end = (pad_W, -pad_W) if pad_W > 0 else (0, W)

    return x[:, h_start:h_end, w_start:w_end, :]


def calculate_fan(x):
    """Calculate fan values for layer initalization."""
    num_input_fmaps, num_output_fmaps = x.shape[-2:]
    receptive_field_size = 1
    if x.ndim > 2:
        receptive_field_size = x[..., 0, 0].size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out
