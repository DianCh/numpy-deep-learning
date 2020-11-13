import numpy as np


def to_one_hot(y, num_class):
    m = len(y)
    y_one_hot = np.zeros((m, num_class))
    y_one_hot[np.arange(m), y.astype(np.int)] = 1

    return y_one_hot
