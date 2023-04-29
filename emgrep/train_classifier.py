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
from tqdm import tqdm


class LinearClassificationHead(torch.nn.Module):
    """Architectural functionality for logistic regression"""

    def __init__(self, inputSize: int, outputSize: int):
        """Initializes the Logistic Regression Model.

        Args:
            inputSize (int): embedding size
            outputSize (int): number of classes
        """
        super(LinearClassificationHead, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        # self.softmax = torch.nn.Softmax(axis=1)

    def forward(self, x: torch.tensor):
        """runs a forward pass

        Args:
            x : input tensor
        """
        out = self.linear(x)
        out = torch.softmax(out, axis=1)
        return out


class DownstreamTuner:
    """Fitting & evaluation functionality for the classification head"""

    def __init__(self, n_classes: int, encoding_size: int, lr=1e-1, epochs=100):
        """Initializes the Classification Meta Model.

        Args:
            n_classes (int): Number of action classes
            encoding_size (int): Encoding size (assuming one-dim.)
            lr (float): initial learning rate
            epochs (int): number of epochs to train for
        """
        self.n_classes = n_classes
        self.encoding_size = encoding_size
        self.head = LinearClassificationHead(encoding_size, n_classes)
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, dataloader: DataLoader) -> DataLoader:
        """Fits a linear logistic regression model on the given training set for a fixed number of epochs.

        @ TODO: Do we want the full [train until cvg w.r.t val set] setup?
        Args:
            dataloaders (DataLoader]): a dataloader delivering tuples of type (embed_1d, class_label).
        Returns:
            DonstreamTuner: itself
        """
        self.head.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.head.parameters(), lr=self.lr, weight_decay=0.001
        )  # , momentum=0.1
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        losses = []

        for epoch in tqdm(range(self.epochs), desc="Training Epoch"):
            ep_loss = []
            for x, y in dataloader:
                optimizer.zero_grad()
                # @TODO how do we want to handle sequences? Curr: Just flatten
                x = torch.reshape(x.float().to(self.device), (-1, self.encoding_size))
                y = torch.reshape(y.long().to(self.device), (-1,))
                outputs = self.head(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

                ep_loss.append(loss.item())

            logging.info("epoch {}, loss {}".format(epoch, np.mean(ep_loss)))
            losses.append(np.mean(ep_loss))

        plt.title("training loss")
        plt.plot(losses)
        plt.show()

        return self

    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        """Parse command line arguments.
        Args:
            dataloader (DataLoader]): a dataloader delivering tuples of type (embed_1d, class_label). (batched!)
        Returns:
            torch.Tensor: Tensor of shape (nsamples, nclasses) with class probabilities.
        """
        with torch.no_grad():
            self.head.to(self.device)
            return torch.cat(
                [
                    self.head(torch.reshape(x.float().to(self.device), (-1, self.encoding_size)))
                    for x, y in dataloader
                ],
                dim=0,
            )

    def score(self, dataloader: DataLoader) -> dict[str, float]:
        """Parse command line arguments.
        Args:
            dataloader (DataLoader]): a dataloader delivering tuples of type (embed_1d, class_label).
        Returns:
            dict[str, float]: test metrics by name. Example: res["accuracy"].
        """
        pred = self.predict(dataloader)
        # @TODO how do we want to handle sequences? Curr: Just flatten
        y = torch.flatten(torch.cat([_y for x, _y in dataloader], dim=0))
        return {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes)(
                torch.argmax(pred, axis=1), y.long()
            ).item(),
            "roc_auc": torchmetrics.AUROC(task="multiclass", num_classes=self.n_classes)(
                pred, y.long()
            ).item(),
            "f1": torchmetrics.F1Score(task="multiclass", num_classes=self.n_classes)(
                torch.argmax(pred, axis=1), y.long()
            ).item(),
        }


def train_classifier(
    representations: dict[str, Dataset], dataloaders: dict[str, DataLoader], args: Namespace
):
    """Train the linear classifier.

    Args:
        representations (dict[str, DataLoader]): Dictionary of representations datasets.
        dataloaders (dict[str, DataLoader]): Dictionary of dataloaders.
        args (Namespace): Command line arguments.
    """
    logging.info("Training the classifier...")
    start = time.time()

    tuner = DownstreamTuner(
        n_classes=args.n_classes, encoding_size=args.ar_dim, epochs=args.epochs_classifier
    )

    tuner.fit(representations["train"])
    # y_pred = np.argmax(tuner.predict("test_dl").numpy(), axis=1)

    res = {
        "train": tuner.score(dataloader=representations["train"]),
        "val": tuner.score(dataloader=representations["val"]),
        "test": tuner.score(dataloader=representations["test"]),
    }

    logging.info("Classification results:")
    logging.info(str(res))

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
    """Runs a testing script with dummy data & plots for the logistic regression head"""
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


# if __name__ == "__main__":
#    test_logreg()
