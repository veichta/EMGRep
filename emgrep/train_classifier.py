"""Implementation of the training loop for classifier."""

import datetime
import logging
import os
import time
from argparse import Namespace
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import wandb
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LinearClassificationHead(torch.nn.Module):
    """Architectural functionality for logistic regression."""

    def __init__(self, input_size: int, output_size: int):
        """Initializes the Classification Head.

        Args:
            input_size (int): Input size.
            output_size (int): Output size.
        """
        super(LinearClassificationHead, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        # self.softmax = torch.nn.Softmax(axis=1)

    def forward(self, x: torch.tensor):
        """Forward pass.

        Args:
            x (torch.tensor): Input tensor.
        """
        out = self.linear(x)
        out = torch.softmax(out, axis=1)
        return out


class DownstreamTuner:
    """Fitting & evaluation functionality for the classification head."""

    def __init__(
        self,
        n_classes: int,
        encoding_size: int,
        args: Namespace,
        lr: float = 1e-1,
        epochs: int = 100,
    ):
        """Initializes the Classification Meta Model.

        Args:
            n_classes (int): Number of action classes.
            encoding_size (int): Encoding size (assuming one-dim).
            args (Namespace): Arguments from the command line.
            lr (float): Initial learning rate.
            epochs (int): Number of epochs to train for.
        """
        self.n_classes = n_classes
        self.encoding_size = encoding_size
        self.head = LinearClassificationHead(encoding_size, n_classes)
        self.lr = lr
        self.epochs = epochs
        self.device = args.device
        self.args = args

    def fit(self, dataloader: DataLoader) -> DataLoader:
        """Fits a linear log-reg model on the given training set for a fixed number of epochs.

        Args:
            dataloader (DataLoader): A dataloader delivering tuples of type (embed_1d, class_label).

        Returns:
            DataLoader: The fitted dataloader.
        """
        # TODO: Do we want the full [train until cvg w.r.t val set] setup?
        self.head.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.head.parameters(), lr=self.lr, weight_decay=0.001
        )  # , momentum=0.1
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        losses = []

        pbar = tqdm(range(self.epochs), desc="Training Epoch")

        for _ in pbar:
            ep_loss = []
            for x, y in dataloader:
                optimizer.zero_grad()
                # @TODO how do we want to handle sequences? Curr: Just flatten
                x = torch.reshape(x.to(self.device), (-1, self.encoding_size))
                y = torch.reshape(y.long().to(self.device), (-1,))
                outputs = self.head(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

                ep_loss.append(loss.item())

            pbar.set_postfix({"loss": np.mean(ep_loss)})
            losses.append(np.mean(ep_loss))

        plt.title("training loss")
        plt.plot(losses, label="loss")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()

        plt.savefig(os.path.join(self.args.log_dir, "classification_loss.png"))
        plt.close()

        return self

    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        """Predicts the class labels for the given dataloader.

        Args:
            dataloader (DataLoader): A dataloader delivering tuples of type (embed_1d, class_label).

        Returns:
            torch.Tensor: Predicted class labels.
        """
        with torch.no_grad():
            self.head.to(self.device)
            return torch.cat(
                [
                    self.head(torch.reshape(x.to(self.device), (-1, self.encoding_size)))
                    for x, y in dataloader
                ],
                dim=0,
            )

    def score(self, dataloader: DataLoader, split: str) -> Dict[str, float]:
        """Computes the test metrics for the given dataloader.

        Args:
            dataloader (DataLoader): A dataloader delivering tuples of type (embed_1d, class_label).
            split (str): The split name.

        Returns:
            Dict[str, float]: Test metrics by name.
        """
        roc_fn = torchmetrics.AUROC(task="multiclass", num_classes=self.n_classes).cpu()

        pred = self.predict(dataloader).cpu()
        # TODO: how do we want to handle sequences? Curr: Just flatten
        y = torch.flatten(torch.cat([_y for x, _y in dataloader], dim=0)).cpu().long()

        pred_classes = torch.argmax(pred, axis=1)
        rep_str = classification_report(
            y.numpy(), pred_classes.numpy(), zero_division=0, output_dict=False
        )
        logging.info(f"{split.capitalize()} classification report:\n{rep_str}")

        report = classification_report(
            y.numpy(), pred_classes.numpy(), output_dict=True, zero_division=0
        )
        report = cleanup_report(report)
        report["roc-auc"] = roc_fn(pred, y).item()
        return report


def cleanup_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans up the classification report.

    Args:
        report (Dict): Classification report.

    Returns:
        Dict: Cleaned up classification report.
    """
    cleaned_report = {}

    for class_label, metrics in report.items():
        if class_label in [str(i) for i in range(12)]:  # class labels
            prefix = f"class-{class_label}-"
            for metric, score in metrics.items():
                cleaned_report[prefix + metric] = score
        elif class_label in ["macro avg", "weighted avg"]:
            prefix = f"{class_label.replace(' ', '-')}-"
            for metric, score in metrics.items():
                cleaned_report[prefix + metric] = score
        elif class_label == "accuracy":
            cleaned_report[class_label] = metrics

    return cleaned_report


def train_classifier(
    representations: Dict[str, Dataset], dataloaders: Dict[str, DataLoader], args: Namespace
):
    """Train the linear classifier.

    Args:
        representations (Dict[str, DataLoader]): Dictionary of representations datasets.
        dataloaders (Dict[str, DataLoader]): Dictionary of dataloaders.
        args (Namespace): Command line arguments.
    """
    logging.info("Training the classifier...")
    start = time.time()

    tuner = DownstreamTuner(
        n_classes=args.n_classes,
        encoding_size=args.ar_dim,
        epochs=args.epochs_classifier,
        args=args,
    )

    tuner.fit(representations["train"])
    # y_pred = np.argmax(tuner.predict("test_dl").numpy(), axis=1)

    res = {
        "train": tuner.score(dataloader=representations["train"], split="train"),
        "val": tuner.score(dataloader=representations["val"], split="val"),
        "test": tuner.score(dataloader=representations["test"], split="test"),
    }

    if args.wandb:
        for metric, val in res["train"].items():
            wandb.log({f"linear-train-{metric}": val})

        for metric, val in res["val"].items():
            wandb.log({f"linear-val-{metric}": val})

        for metric, val in res["test"].items():
            wandb.log({f"linear-test-{metric}": val})

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
    """Runs a testing script with dummy data & plots for the logistic regression head."""
    # create dummy data
    encoding_size = 2
    # n_classes = 2

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

    print(f"Train shape: X {x_train.shape} y {y_train.shape}")
    print(f"Test shape: X {x_test.shape} y {y_test.shape}")
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
