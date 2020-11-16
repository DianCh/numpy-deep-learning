import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        print(x.shape)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


def get_data_tensor(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
        imgs = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        imgs = (imgs / 255.0 - 0.5) / 0.5
        labels = np.array(data[b"labels"]).astype(np.int)

    return imgs, labels


# define model
net = Net()

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

start = 0
end = start + batch_size
X_batch, y_batch = X_train[start:end], y_train[start:end]
y_batch_one_hot = to_one_hot(y_batch, num_class)
