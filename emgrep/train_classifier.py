"""Implementation of the training loop for classifier."""

import datetime
import logging
import time
from argparse import Namespace
from typing import Any, Dict

import numpy as np
import torch
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LinearClassificationHead(nn.Module):
    """Architectural functionality for logistic regression."""

    def __init__(self, input_size: int, output_size: int):
        """Initializes the Classification Head.

        Args:
            input_size (int): Input size.
            output_size (int): Output size.
        """
        super(LinearClassificationHead, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.tensor):
        """Forward pass.

        Args:
            x (torch.tensor): Input tensor.
        """
        out = self.ffn(x)
        out = self.softmax(out)
        return out


def train_classifier(representations: Dict[str, Dataset], pred_block: int, args: Namespace):
    """Train the linear classifier.

    Args:
        representations (Dict[str, DataLoader]): Dictionary of representations datasets.
        pred_block (int): Block to predict from sequence.
        args (Namespace): Command line arguments.
    """
    logging.info("Training the classifier...")
    start = time.time()

    n_classes = len(representations["train"].actual_labels)
    logging.debug(f"Number of classes: {n_classes}")

    # define data loaders
    train_dl = DataLoader(representations["train"], batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(representations["val"], batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(representations["test"], batch_size=args.batch_size, shuffle=False)

    # define model
    model = LinearClassificationHead(input_size=args.ar_dim, output_size=n_classes)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_classifier)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5  # , verbose=True
    )

    # train loop
    metrics: Dict[str, Any] = {"train": {}, "val": {}, "test": {}}
    pbar = tqdm(range(args.epochs_classifier), desc="Training classifier", ncols=100)
    for epoch in pbar:
        metrics["train"][epoch] = train_one_epoch_classifier(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_dl=train_dl,
            pred_block=pred_block,
            args=args,
        )
        metrics["val"][epoch] = validate_classifier(
            model=model, criterion=criterion, val_dl=val_dl, pred_block=pred_block, args=args
        )

        scheduler.step(metrics["val"][epoch]["loss"])

        pbar.set_postfix(
            {
                "train loss": metrics["train"][epoch]["loss"],
                "val loss": metrics["val"][epoch]["loss"],
            }
        )

    metrics["test"][0] = test_classifier(
        model=model, criterion=criterion, test_dl=test_dl, pred_block=pred_block, args=args
    )

    # evaluate
    out = {
        "train": report(
            model=model, dataloader=train_dl, pred_block=pred_block, args=args, split="train"
        ),
        "val": report(
            model=model, dataloader=val_dl, pred_block=pred_block, args=args, split="valid"
        ),
        "test": report(
            model=model, dataloader=test_dl, pred_block=pred_block, args=args, split="test"
        ),
    }

    if args.wandb:
        for metric, val in out["train"].items():
            wandb.log({f"linear-train-{metric}": val})

        for metric, val in out["val"].items():
            wandb.log({f"linear-val-{metric}": val})

        for metric, val in out["test"].items():
            wandb.log({f"linear-test-{metric}": val})

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    logging.info(f"Training took {duration}.")

    return out


def train_one_epoch_classifier(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: DataLoader,
    pred_block: int,
    args: Namespace,
) -> Dict[str, Any]:
    """Train the classifier for one epoch.

    Args:
        model (nn.Module): Model.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        train_dl (DataLoader): Train dataloader.
        pred_block (int): Block to predict from sequence.
        args (Namespace): Command line arguments.

    Returns:
        Dict[str, Any]: Dictionary of metrics.
    """
    epoch_loss = 0.0

    model.train()
    model.to(args.device)
    for batch in train_dl:
        # get the inputs and labels
        inputs, labels = batch
        inputs = inputs[:, pred_block]
        labels = labels[:, pred_block]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(args.device))
        loss = criterion(outputs, labels.to(args.device))
        loss.backward()
        optimizer.step()

        # compute the statistics
        epoch_loss += loss.item()

    epoch_loss /= len(train_dl)

    return {"loss": epoch_loss}


def validate_classifier(
    model: nn.Module, criterion: nn.Module, val_dl: DataLoader, pred_block: int, args: Namespace
) -> Dict[str, Any]:
    """Validate the classifier for one epoch.

    Args:
        model (nn.Module): Model.
        criterion (nn.Module): Loss function.
        val_dl (DataLoader): Validation dataloader.
        pred_block (int): Block to predict from sequence.
        args (Namespace): Command line arguments.

    Returns:
        Dict[str, Any]: Dictionary of metrics.
    """
    epoch_loss = 0.0

    model.eval()
    model.to(args.device)
    with torch.no_grad():
        for batch in val_dl:
            # get the inputs and labels
            inputs, labels = batch
            inputs = inputs[:, pred_block]
            labels = labels[:, pred_block]

            # forward
            outputs = model(inputs.to(args.device))
            loss = criterion(outputs, labels.to(args.device))

            # compute the statistics
            epoch_loss += loss.item()

    epoch_loss /= len(val_dl)

    return {"loss": epoch_loss}


def test_classifier(
    model: nn.Module, criterion: nn.Module, test_dl: DataLoader, pred_block: int, args: Namespace
) -> Dict[str, Any]:
    """Test the classifier.

    Args:
        model (nn.Module): Model.
        criterion (nn.Module): Loss function.
        test_dl (DataLoader): Test dataloader.
        pred_block (int): Block to predict from sequence.
        args (Namespace): Command line arguments.

    Returns:
        Dict[str, Any]: Dictionary of metrics.
    """
    epoch_loss = 0.0

    model.eval()
    model.to(args.device)
    with torch.no_grad():
        for batch in test_dl:
            # get the inputs and labels
            inputs, labels = batch
            inputs = inputs[:, pred_block]
            labels = labels[:, pred_block]

            # forward
            outputs = model(inputs.to(args.device))
            loss = criterion(outputs, labels.to(args.device))

            # compute the statistics
            epoch_loss += loss.item()

    epoch_loss /= len(test_dl)

    return {"loss": epoch_loss}


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


def report(
    model: nn.Module,
    dataloader: DataLoader,
    split: str,
    pred_block: int,
    args: Namespace,
) -> Dict[str, float]:
    """Generate classification report and confusion matrix.

    Args:
        model (nn.Module): Classification Model.
        dataloader (DataLoader): Dataloader for the evaluation.
        split (str): Split of the dataloader.
        pred_block (int): Block id where the encoding will be taking for prediction.
        args (Namespace): Command line arguments.

    Returns:
        Dict[str, float]: Classification report.
    """
    model.eval()
    model.to(args.device)

    # get targets and predictions
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs[:, pred_block]
            labels = labels[:, pred_block]

            outputs = model(inputs.to(args.device))
            predictions.append(outputs.argmax(dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    # compute metrics
    rev_label_map = {v: k for k, v in dataloader.dataset.label_map.items()}
    n_classes = len(rev_label_map)
    class_names = [str(rev_label_map[c]) for c in range(n_classes)]

    rep_str = classification_report(
        y_true=targets,
        y_pred=predictions,
        target_names=class_names,
        zero_division=0,
    )
    logging.info(f"{split.capitalize()} classification report:\n{rep_str}")

    report = classification_report(
        y_true=targets,
        y_pred=predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report = cleanup_report(report)

    cf = confusion_matrix(targets, predictions)
    logging.info(f"{split.capitalize()} confusion matrix:\n{cf}")

    return report
