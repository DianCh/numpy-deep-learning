"""Prepare MNIST and CIFAR10 datasets for the experiments."""
import os
import shutil
import wget

import numpy as np
from sklearn.datasets import fetch_openml


def get_mnist(data_root):
    """Get MNIST files."""
    # download
    print("Downloading MNIST")
    X, y = fetch_openml(
        name="mnist_784",
        version=1,
        return_X_y=True,
        data_home="tmp",
        as_frame=False,
    )

    # dump to disk
    print("Unpacking MNIST")
    mnist_dir = os.path.join(data_root, "mnist-784")
    os.makedirs(mnist_dir, exist_ok=True)

    with open(os.path.join(mnist_dir, "images.npy"), "wb") as fh:
        np.save(fh, X)

    with open(os.path.join(mnist_dir, "labels.npy"), "wb") as fh:
        np.save(fh, y)

    # remove cache
    shutil.rmtree("tmp")


def get_cifar10(data_root):
    """Get CIFAR10 files."""
    # download
    print("Downloading CIFAR10")
    data_path = wget.download(
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", data_root
    )

    # unpack archive
    print("\nUnpacking CIFAR10...")
    os.system(f"tar -xf {data_path} -C {data_root}")

    # remove archive
    os.remove(data_path)


if __name__ == "__main__":
    data_root = "data"
    shutil.rmtree(data_root)
    os.makedirs(data_root, exist_ok=True)

    get_mnist(data_root)
    get_cifar10(data_root)
