"""Implementation of the training loop for classifier."""

import datetime
import logging
import time
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader, Dataset


class LinearClassificationHead(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearClassificationHead, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        # self.softmax = torch.nn.Softmax(axis=1)

    def forward(self, x):
        out = self.linear(x)
        out = torch.softmax(out, axis=1)
        return out


class DownstreamTuner:
    def __init__(self, n_classes, encoding_size, lr=1e-1, epochs=100):
        self.n_classes = n_classes
        self.head = LinearClassificationHead(encoding_size, n_classes)
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, dataloader):
        self.head.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.head.parameters(), lr=self.lr, weight_decay=0.001
        )  # , momentum=0.1
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        losses = []

        for epoch in range(self.epochs):
            ep_loss = 0
            for i, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.head(x.float().to(self.device))
                loss = criterion(outputs, y.long().to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()

                ep_loss += loss.item()

            logging.info("epoch {}, loss {}".format(epoch, ep_loss / i))
            losses.append(ep_loss / i)

        plt.title("training loss")
        plt.plot(losses)
        plt.show()

        return self

    def predict(self, dataloader):
        with torch.no_grad():
            self.head.to(self.device)
            return torch.cat([self.head(x.float().to(self.device)) for x, y in dataloader], dim=0)

    def score(self, dataloader):
        pred = self.predict(dataloader)
        y = torch.cat([_y for x, _y in dataloader], dim=0)
        scores = {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes)(
                torch.argmax(pred, axis=1), y.long()
            ).item(),
            "roc_auc": torchmetrics.AUROC(task="multiclass", num_classes=self.n_classes)(
                pred, y.long()
            ).item(),
        }
        logging.info(scores)
        return scores


def train_classifier(
    representations: dict[str, Dataset], dataloaders: dict[str, DataLoader], args: Namespace
):
    """Train the classifier.

    Args:
        representations (dict[str, DataLoader]): Dictionary of representations datasets.
        dataloaders (dict[str, DataLoader]): Dictionary of dataloaders.
        args (Namespace): Command line arguments.
    """
    logging.info("Training the classifier...")
    start = time.time()

    # TODO: Initialize
    # model = None
    # criterion = None

    print(args)
    tuner = DownstreamTuner(
        n_classes=args.n_classes, encoding_size=args.encoding_size, epochs=args.epochs
    )

    # TODO: Train classifier

    # metrics = {"train": {}, "val": {}, "test": {}}
    # for epoch in range(args.epochs_classifier):
    #     train_one_epoch_classifier(...)
    #     validate_classifier(...)
    #     save_checkpoint_classifier(...)

    # test_classifier(...)

    end = time.time()
    elapsed = datetime.timedelta(seconds=end - start)
    logging.info(f"Training time: {elapsed}")

    # TODO: log metrics


def test_logreg():
    # create dummy data
    encoding_size = 2
    n_classes = 2

    x_train = np.array([np.arange(0, 100) + 10 * d for d in range(encoding_size)])
    x_train = x_train + 100 * np.random.rand(*x_train.shape)
    x_train = x_train.swapaxes(0, 1) / 100

    y_train = np.concatenate([np.zeros(len(x_train) // 2), np.ones(len(x_train) // 2)], axis=0)

    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=10,
        shuffle=True,
    )

    x_test = np.array([0.5 + np.arange(0, 100) + 10 * d for d in range(encoding_size)])
    x_test = x_test + 100 * np.random.rand(*x_test.shape)
    x_test = x_test.swapaxes(0, 1) / 100

    y_test = np.concatenate([np.zeros(len(x_train) // 2), np.ones(len(x_train) // 2)], axis=0)

    test_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=10
    )

    print("Train shape: X {} y {}".format(x_train.shape, y_train.shape))
    print("Test shape: X {} y {}".format(x_test.shape, y_test.shape))
    print("Dataloaders: ", list(train_dl)[0], list(test_dl)[0])

    tuner = DownstreamTuner(2, encoding_size)
    tuner.fit(train_dl)
    y_pred = np.argmax(tuner.predict(test_dl).numpy(), axis=1)

    res = tuner.score(test_dl)
    print("FINAL TEST SCORE: ", res)

    plt.subplot(1, 3, 1)
    plt.title("training ground truth")
    plt.scatter(x_train[y_train.flatten() == 0, :][:, 0], x_train[y_train.flatten() == 0, :][:, 1])
    plt.scatter(x_train[y_train.flatten() == 1, :][:, 0], x_train[y_train.flatten() == 1, :][:, 1])
    plt.legend(["Class 0", "Class 1"])

    plt.subplot(1, 3, 2)
    plt.title("test ground truth")
    plt.scatter(x_test[y_test.flatten() == 0, :][:, 0], x_test[y_test.flatten() == 0, :][:, 1])
    plt.scatter(x_test[y_test.flatten() == 1, :][:, 0], x_test[y_test.flatten() == 1, :][:, 1])
    plt.legend(["Class 0", "Class 1"])

    plt.subplot(1, 3, 3)
    plt.title("test pred (acc={:.2f})".format(res["accuracy"]))
    plt.scatter(x_test[y_pred.flatten() == 0, :][:, 0], x_test[y_pred.flatten() == 0, :][:, 1])
    plt.scatter(x_test[y_pred.flatten() == 1, :][:, 0], x_test[y_pred.flatten() == 1, :][:, 1])
    plt.legend(["Class 0", "Class 1"])

    plt.show()


if __name__ == "__main__":
    test_logreg()
