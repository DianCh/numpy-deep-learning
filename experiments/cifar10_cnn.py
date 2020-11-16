from math import ceil
import numpy as np
import pickle
import os

from ndl.model import SimpleConvNet
from ndl.loss import CrossEntropyLoss
from ndl.optimizer import SGD
from ndl.utils import to_one_hot


def get_data_tensor(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
        imgs = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        imgs = (imgs / 255.0 - 0.5) / 0.5
        labels = np.array(data[b"labels"]).astype(np.int)

    return imgs, labels


def run():
    # define model
    net = SimpleConvNet()

    # define loss
    loss_fn = CrossEntropyLoss()

    # prepare data
    num_train, num_class = 50000, 10
    data_root = "data/cifar10"
    X_train, y_train = [], []
    for i in range(1, 6):
        train_file = os.path.join(data_root, f"data_batch_{i}")
        imgs, labels = get_data_tensor(train_file)
        X_train.append(imgs)
        y_train.append(labels)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    test_file = os.path.join(data_root, "test_batch")
    X_test, y_test = get_data_tensor(test_file)
    print(
        "Loaded {} samples for training, {} samples for testing".format(
            len(X_train), len(X_test)
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
