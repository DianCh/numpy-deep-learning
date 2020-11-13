from math import ceil
from sklearn.datasets import fetch_openml
import numpy as np

from ndl.model import MultiLayerPerceptron
from ndl.loss import CrossEntropyLoss
from ndl.optimizer import SGD
from ndl.utils import to_one_hot


def run():
    # define model
    in_feature, out_feature = 784, 10
    hidden = [50, 20]
    net = MultiLayerPerceptron(in_feature, out_feature, hidden)

    # define loss
    loss_fn = CrossEntropyLoss()

    # prepare data
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, data_home="data")
    X /= 255.0
    y = y.astype(np.int)
    num_train, num_class = 60000, 10
    X_train, y_train = X[:num_train], y[:num_train]
    X_test, y_test = X[num_train:], y[num_train:]
    print(
        "Loaded {} examples. {} for training, {} for testing".format(
            len(X), len(X_train), len(X_test)
        )
    )

    # define training hyper-parameters
    batch_size = 100
    learning_rate = 0.05
    num_epoch = 20
    stat_every = 20

    # define optimizer
    optimizer = SGD(net, lr=learning_rate)

    # start training
    for epoch in range(num_epoch):
        for batch in range(ceil(num_train / batch_size)):
            # slice batch data
            start = batch * batch_size
            end = start + batch_size
            X_batch, y_batch = X_train[start:end], y_train[start:end]
            y_batch_one_hot = to_one_hot(y_batch, num_class)

            optimizer.zero_grad()

            # model output
            logits = net.forward(X_batch)
            pred = np.argmax(logits, axis=1)

            # cross-entropy loss
            ce_loss = loss_fn.forward(logits, y_batch_one_hot)
            gradient = loss_fn.backward()

            optimizer.step(gradient)

            train_acc = np.mean(pred == y_batch)
            if (batch + 1) % stat_every == 0:
                print(
                    "Epoch {}, Batch {}, loss: {}, training accuracy: {:.3f}".format(
                        epoch, batch, ce_loss, train_acc
                    )
                )

        # run evaluation on test split after each epoch
        logits = net.forward(X_test)
        pred = np.argmax(logits, axis=1)
        val_acc = np.mean(pred == y_test)
        print("-----")
        print(f"Epoch {epoch}, test accuracy: {val_acc:.3f}")


if __name__ == "__main__":
    run()
