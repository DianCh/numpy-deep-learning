"""Utilities that help manipulate tensors."""
import os
import pickle

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


def to_one_hot(y, num_class):
    """Convert [0, m) ranged labels to one-hot representation."""
    m = len(y)
    y_one_hot = np.zeros((m, num_class))
    y_one_hot[np.arange(m), y.astype(np.int)] = 1

    return y_one_hot


def calculate_fan(x):
    """Calculate fan values for layer initalization."""
    num_input_fmaps, num_output_fmaps = x.shape[-2:]
    receptive_field_size = 1
    if x.ndim > 2:
        receptive_field_size = x[..., 0, 0].size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def get_data_tensor(data_path):
    """Read in one batch of CIFAR10 file."""
    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
        imgs = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        imgs = (imgs / 255.0 - 0.5) / 0.5
        labels = np.array(data[b"labels"]).astype(np.int)

    return imgs, labels


def load_cifar10_data(data_root):
    """Load CIFAR10 data files into np arrays."""
    # train split
    X_train, y_train = [], []
    for i in range(1, 6):
        train_file = os.path.join(data_root, f"data_batch_{i}")
        imgs, labels = get_data_tensor(train_file)
        X_train.append(imgs)
        y_train.append(labels)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # test split
    test_file = os.path.join(data_root, "test_batch")
    X_test, y_test = get_data_tensor(test_file)
    print(
        f"Loaded {len(X_train) + len(X_test)} samples.",
        f"{len(X_train)} for training,",
        f"{len(X_test)} for testing",
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_mnist_data(data_root, num_train):
    """Load MNIST data files into np arrays."""
    with open(os.path.join(data_root, "images.npy"), "rb") as fh:
        X = np.load(fh)

    with open(os.path.join(data_root, "labels.npy"), "rb") as fh:
        y = np.load(fh, allow_pickle=True)

    X = (X / 255.0 - 0.5) / 0.5
    y = y.astype(np.int)
    X_train, y_train = X[:num_train], y[:num_train]
    X_test, y_test = X[num_train:], y[num_train:]
    print(
        f"Loaded {len(X)} samples.",
        f"{len(X_train)} for training,",
        f"{len(X_test)} for testing",
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
