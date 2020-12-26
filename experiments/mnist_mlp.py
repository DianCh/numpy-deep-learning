"""Script for Multi-Layer Perceptron experiment."""
import sys

from ndl.model import MultiLayerPerceptron
from ndl.trainer import MultiClassTrainer
from ndl.utils import load_mnist_data


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
        "stat_every": 20,
        "show_curve": True,
        "class_names": [f"digit {i}" for i in range(10)],
    }
    run(sys.argv[1], kwargs)
