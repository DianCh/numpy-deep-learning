"""Script for Convolutional Neural Network experiment."""
import pickle
import os
import sys
import numpy as np


from ndl.model import SimpleConvNet
from ndl.trainer import MultiClassTrainer


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


def run(data_root, kwargs):
    """Main logic for train & evaluation."""
    # define model
    net = SimpleConvNet()

    # prepare data
    data = load_cifar10_data(data_root)

    trainer = MultiClassTrainer(net, data, **kwargs)
    trainer.train()


if __name__ == "__main__":
    kwargs = {
        "num_class": 10,
        "batch_size": 16,
        "num_epoch": 10,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "stat_every": 20,
        "show_curve": True,
        "class_names": [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    }
    run(sys.argv[1], kwargs)
