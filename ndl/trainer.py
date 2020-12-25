"""Trainers that contain the main train/eval loop."""
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from ndl.loss import CrossEntropyLoss
from ndl.optimizer import SGD
from ndl.utils import to_one_hot


class MultiClassTrainer:
    """Trainer for multi-class classifier."""

    def __init__(
        self,
        model,
        data,
        num_class,
        batch_size,
        num_epoch,
        shuffle=True,
        learning_rate=0.0005,
        momentum=0.9,
        stat_every=20,
        show_curve=False,
        **kwargs,
    ):
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]
        self.num_class = num_class
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.shuffle = shuffle
        self.stat_every = stat_every
        self.show_curve = show_curve

        self.loss_fn = CrossEntropyLoss()
        self.optimizer = SGD(model, lr=learning_rate, momentum=momentum)

    def train(self):
        """Main loop of training."""
        num_batch = ceil(len(self.X_train) / self.batch_size)
        losses = []
        train_accs = []
        val_accs = [self.evaluate()]
        for epoch in range(self.num_epoch):
            if self.shuffle:
                idx = np.random.permutation(len(self.X_train))
                self.X_train = self.X_train[idx, :]
                self.y_train = self.y_train[idx]

            for batch in range(num_batch):
                # slice batch data
                start = batch * self.batch_size
                end = start + self.batch_size
                X_batch, y_batch = (
                    self.X_train[start:end],
                    self.y_train[start:end],
                )
                y_batch_one_hot = to_one_hot(y_batch, self.num_class)

                self.optimizer.zero_grad()

                # model output
                logits = self.model.forward(X_batch)
                pred = np.argmax(logits, axis=1)

                # cross-entropy loss
                ce_loss = self.loss_fn.forward(logits, y_batch_one_hot)

                # back propagation and update
                gradient = self.loss_fn.backward()
                self.model.backward(gradient)
                self.optimizer.step()

                # evaluate this batch
                train_acc = np.mean(pred == y_batch)
                if (batch + 1) % self.stat_every == 0:
                    losses.append(ce_loss)
                    train_accs.append(train_acc)
                    print(
                        f"Epoch {epoch},",
                        f"Batch {batch},",
                        f"loss: {ce_loss},",
                        f"training accuracy: {train_acc}",
                    )

            # run evaluation on test split after each epoch
            val_acc = self.evaluate()
            val_accs.append(val_acc)
            print("-----")
            print(f"Test accuracy after {epoch + 1} epochs: {val_acc:.3f}")

        if self.show_curve:
            self.draw_curves(losses, train_accs, val_accs, num_batch)

    def evaluate(self):
        """Get accuracy on validation set."""
        logits = self.model.forward(self.X_test)
        pred = np.argmax(logits, axis=1)
        val_acc = np.mean(pred == self.y_test)

        return val_acc

    def draw_curves(self, losses, train_accs, val_accs, num_batch):
        """Draw loss & accuracy curves of train & evaluation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # loss curve
        ax1.plot(
            range(0, self.num_epoch * num_batch, self.stat_every),
            losses,
        )
        ax1.set_title("Loss Curve")
        ax1.set(xlabel="steps", ylabel="loss")

        # accuracy curves
        ax2.plot(
            range(0, self.num_epoch * num_batch, self.stat_every),
            train_accs,
            "b",
            label="train",
        )
        ax2.plot(
            range(0, self.num_epoch * num_batch + 1, num_batch),
            val_accs,
            "r",
            label="val",
        )
        ax2.set_title("Accuracy Curves")
        ax2.set(xlabel="steps", ylabel="accuracy")
        ax2.legend(loc="lower right")
        fig.savefig("metrics.png")
