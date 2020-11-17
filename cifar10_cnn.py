from math import ceil
import numpy as np
import pickle
import os

from ndl.model import SimpleConvNet
from ndl.loss import CrossEntropyLoss
from ndl.optimizer import SGD
from ndl.utils import to_one_hot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time


class Net(nn.Module):
    def __init__(self, ndl_net):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        with torch.no_grad():
            self._reset(self.conv1, ndl_net.layers[0])
            self._reset(self.conv2, ndl_net.layers[3])
            self._reset(self.fc1, ndl_net.layers[7], False)
            self._reset(self.fc2, ndl_net.layers[9], False)
            self._reset(self.fc3, ndl_net.layers[11], False)

    def _reset(self, layer, ndl_layer, conv=True):
        print(layer.weight.shape, "---", ndl_layer.W.shape)
        if conv:
            layer.weight = nn.Parameter(
                torch.from_numpy(ndl_layer.W.transpose(3, 2, 0, 1).copy())
            )
        else:
            layer.weight = nn.Parameter(
                torch.from_numpy(ndl_layer.W.transpose().copy())
            )
        layer.bias = nn.Parameter(torch.from_numpy(ndl_layer.b.copy()))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
    net_torch = Net(net).double()

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
    batch_size = 8
    learning_rate = 0.001
    num_epoch = 20
    stat_every = 1

    # define optimizer
    optimizer = SGD(net, lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    optimizer_torch = optim.SGD(net_torch.parameters(), lr=learning_rate, momentum=0.9)

    # start training
    for epoch in range(num_epoch):
        for batch in range(ceil(num_train / batch_size)):
            # slice batch data
            start = batch * batch_size
            end = start + batch_size
            X_batch, y_batch = X_train[start:end], y_train[start:end]
            y_batch_one_hot = to_one_hot(y_batch, num_class)

            # optimizer_torch.zero_grad()

            # y_torch = torch.tensor(y_batch)
            # outputs = net_torch(torch.from_numpy(X_batch.transpose(0, 3, 1, 2)))
            # loss = criterion(outputs, y_torch)
            # loss.backward()
            # optimizer_torch.step()

            # pred = torch.max(outputs, dim=1)
            # # print(pred.indices)
            # train_acc = torch.mean((pred.indices == y_torch).float())

            # if (batch + 1) % stat_every == 0:
            #     print(
            #         "Pytorch Epoch {}, Batch {}, loss: {}, training accuracy: {:.3f}".format(
            #             epoch, batch, loss.detach().numpy(), train_acc
            #         )
            #     )
            # print(outputs)

            # return
            optimizer.zero_grad()

            # model output
            logits = net.forward(X_batch)
            pred = np.argmax(logits, axis=1)

            # cross-entropy loss
            ce_loss = loss_fn.forward(logits, y_batch_one_hot)
            gradient = loss_fn.backward()
            net.backward(gradient)

            optimizer.step()

            train_acc = np.mean(pred == y_batch)
            if (batch + 1) % stat_every == 0:
                print(
                    "NDL Epoch {}, Batch {}, loss: {}, training accuracy: {:.3f}".format(
                        epoch, batch, ce_loss, train_acc
                    )
                )
            # print(logits)
            # print("-----")
        # run evaluation on test split after each epoch
        logits = net.forward(X_test)
        pred = np.argmax(logits, axis=1)
        val_acc = np.mean(pred == y_test)
        print("-----")
        print(f"Epoch {epoch}, test accuracy: {val_acc:.3f}")
        # with torch.no_grad():
        #     outputs = net_torch(torch.from_numpy(X_test.transpose((0, 3, 1, 2))))
        #     pred = torch.max(outputs, dim=1)
        #     # print(pred.indices)
        #     y_test = torch.tensor(y_test)
        #     test_acc = torch.mean((pred.indices == y_test).float()).numpy()
        #     print(f"Epoch {epoch}, test accuracy: {test_acc:.3f}")
        #     time.sleep(1.0)


if __name__ == "__main__":
    run()
