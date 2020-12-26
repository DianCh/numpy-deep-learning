"""Script for Convolutional Neural Network experiment."""
import sys

from ndl.model import SimpleConvNet
from ndl.trainer import MultiClassTrainer
from ndl.utils import load_cifar10_data


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
