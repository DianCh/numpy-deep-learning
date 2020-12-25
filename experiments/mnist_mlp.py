"""Script for Multi-Layer Perceptron experiment."""
import os
import sys
import numpy as np

from ndl.model import MultiLayerPerceptron
from ndl.trainer import MultiClassTrainer


def load_mnist_data(data_root, num_train):
    """Load mnist data files into np arrays."""
    with open(os.path.join(data_root, "images.npy"), "rb") as fh:
        X = np.load(fh)

    with open(os.path.join(data_root, "labels.npy"), "rb") as fh:
        y = np.load(fh)

    X = (X / 255.0 - 0.5) / 0.5
    y = y.astype(np.int)
    X_train, y_train = X[:num_train], y[:num_train]
    X_test, y_test = X[num_train:], y[num_train:]
    print(
        "Loaded {} samples. {} for training, {} for testing".format(
            len(X), len(X_train), len(X_test)
        )
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def run(data_root, kwargs):
    """Main logic for train & evaluation."""
    # define model
    in_feature, out_feature = 784, 10
    hidden = [50, 20]
    net = MultiLayerPerceptron(in_feature, out_feature, hidden)

    # prepare data
    data = load_mnist_data(data_root, kwargs["num_train"])

    trainer = MultiClassTrainer(net, data, **kwargs)
    trainer.train()


if __name__ == "__main__":
    kwargs = {
        "num_train": 60000,
        "num_class": 10,
        "batch_size": 100,
        "num_epoch": 10,
        "learning_rate": 0.0005,
        "momentum": 0.9,
        "stat_every": 100,
        "show_curve": True,
    }
    run(sys.argv[1], kwargs)
