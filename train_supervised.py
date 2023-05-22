"""Train supervised model on EMGRep dataset."""

import datetime
import logging
import os
import time
from argparse import Namespace
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from emgrep.datasets.EMGRepDataloader import EMGRepDataloader, get_split
from emgrep.models.cpc_model import CPCAR, CPCEncoder
from emgrep.utils.utils import cleanup, setup


class SupervisedModel(nn.Module):
    """Supervised model."""

    def __init__(self, encoder: CPCEncoder, ar: CPCAR, args: Namespace):
        """Initialize.

        Args:
            encoder (CPCEncoder): Encoder model.
            ar (CPCAR): Autoregressive model.
            args (Namespace): Command line arguments.
        """
        super().__init__()
        self.gEnc = encoder
        self.gAR = ar

        self.ffn = nn.Sequential(
            nn.Linear(args.ar_dim, args.n_classes),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass."""
        z = self.gEnc(x)
        c = self.gAR(z)
        return self.ffn(c[:, 0, -1, :])


def train_one_epoch(
    model: SupervisedModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: Namespace,
) -> Dict[str, Any]:
    """Train one epoch.

    Args:
        model (SupervisedModel): The model to train.
        dataloader (DataLoader): The dataloader to use.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): The current epoch.
        args (Namespace): The command line arguments.

    Returns:
        Dict[str, Any]: The metrics.
    """
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch} / {args.epochs_cpc}", ncols=100)
    losses = []
    model.train()
    model.to(args.device)
    targets = []
    preds = []
    for x, y, _ in pbar:
        labels = y[:, 0, -1, -1, 0]  # last label
        optimizer.zero_grad()

        out = model(x.to(args.device))

        loss = criterion(out, labels.long().to(args.device))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if args.wandb:
            wandb.log({"supervised_train_loss": loss.item()})

        pbar.set_postfix({"loss": np.mean(losses)})

        targets.append(labels.cpu().detach().numpy())
        preds.append(out.cpu().detach().numpy())

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    preds = np.argmax(preds, axis=1)

    classes = np.unique(targets).astype(int)
    report = classification_report(
        targets, preds, labels=classes, output_dict=True, zero_division=0
    )
    rep_str = classification_report(targets, preds, labels=classes, zero_division=0)
    report = cleanup_report(report)

    logging.info(f"Train Classification Report:\n{rep_str}")

    report["loss"] = np.mean(losses)

    if args.wandb:
        for metric, val in report.items():
            wandb.log({f"supervised-train-{metric}": val})

    return report


def validate(
    model: SupervisedModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    args: Namespace,
) -> Dict[str, Any]:
    """Validate one epoch.

    Args:
        model (SupervisedModel): The model to validate.
        dataloader (DataLoader): The dataloader to use.
        criterion (nn.Module): The loss function.
        epoch (int): The current epoch.
        args (Namespace): The command line arguments.

    Returns:
        Dict[str, Any]: The metrics.
    """
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch} / {args.epochs_cpc}", ncols=100)
    losses = []
    model.eval()
    model.to(args.device)
    targets = []
    preds = []
    with torch.no_grad():
        for x, y, _ in pbar:
            labels = y[:, 0, -1, -1, 0]  # last label
            out = model(x.to(args.device))
            loss = criterion(out, labels.long().to(args.device))

            losses.append(loss.item())
            if args.wandb:
                wandb.log({"supervised_val_loss": loss.item()})

            pbar.set_postfix({"loss": np.mean(losses)})

            targets.append(labels.cpu().detach().numpy())
            preds.append(out.cpu().detach().numpy())

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    preds = np.argmax(preds, axis=1)

    classes = np.unique(targets).astype(int)
    report = classification_report(
        targets, preds, labels=classes, output_dict=True, zero_division=0
    )
    rep_str = classification_report(targets, preds, labels=classes, zero_division=0)
    report = cleanup_report(report)

    logging.info(f"Validation Classification Report:\n{rep_str}")

    report["loss"] = np.mean(losses)

    if args.wandb:
        for metric, val in report.items():
            wandb.log({f"supervised-val-{metric}": val})

    return report


def test(
    model: SupervisedModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    args: Namespace,
) -> Dict[str, Any]:
    """Test one epoch.

    Args:
        model (SupervisedModel): The model to test.
        dataloader (DataLoader): The dataloader to use.
        criterion (nn.Module): The loss function.
        epoch (int): The current epoch.
        args (Namespace): The command line arguments.

    Returns:
        Dict[str, Any]: The metrics.
    """
    pbar = tqdm(dataloader, desc=f"Test Epoch {epoch} / {args.epochs_cpc}", ncols=100)
    losses = []
    model.eval()
    model.to(args.device)
    targets = []
    preds = []
    with torch.no_grad():
        for x, y, _ in pbar:
            labels = y[:, 0, -1, -1, 0]
            out = model(x.to(args.device))
            loss = criterion(out, labels.long().to(args.device))

            losses.append(loss.item())
            if args.wandb:
                wandb.log({"supervised_test_loss": loss.item()})
            pbar.set_postfix({"loss": np.mean(losses)})

            targets.append(labels.cpu().detach().numpy())
            preds.append(out.cpu().detach().numpy())

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    preds = np.argmax(preds, axis=1)

    classes = np.unique(targets).astype(int)
    report = classification_report(
        targets, preds, labels=classes, output_dict=True, zero_division=0
    )
    rep_str = classification_report(targets, preds, labels=classes, zero_division=0)
    report = cleanup_report(report)

    logging.info(f"Test Classification Report:\n{rep_str}")

    report["loss"] = np.mean(losses)

    if args.wandb:
        for metric, val in report.items():
            wandb.log({f"supervised-test-{metric}": val})

    return report


def save_checkpoint(
    model: SupervisedModel, epoch: int, args: Namespace, metrics: List[Dict[str, Any]]
) -> None:
    """Save the model checkpoint.

    Args:
        model (SupervisedModel): The model to save.
        epoch (int): The current epoch.
        args (Namespace): The command line arguments.
        metrics (List[Dict[str, Any]]): The metrics.
    """
    model_dir = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    losses = [m["loss"] for m in metrics]
    best_epoch = np.argmin(losses) + 1

    if epoch == best_epoch:
        logging.info(f"Saving model checkpoint at epoch {epoch+1}...")
        torch.save(model.to("cpu").state_dict(), os.path.join(model_dir, "best_model.pt"))
        model.to(args.device)


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
        else:
            logging.warning(f"Unknown class label {class_label} ({type(class_label)})")

    return cleaned_report


def main(args: Namespace):
    """Main function.

    Args:
        args (Namespace): Command line arguments.
    """
    start = time.time()
    timestep = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join(
        *args.log_dir.split("/")[:-2], f"supervised_{args.val_idx}_{args.test_idx}", timestep
    )
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Load data
    train_split, val_split, test_split = get_split(args)

    dl = EMGRepDataloader(
        data_path=args.data,
        train_data=train_split,
        val_data=val_split,
        test_data=test_split,
        positive_mode=args.positive_mode,
        seq_len=args.seq_len,
        seq_stride=args.block_len,  # predict label of last block
        block_len=args.block_len,
        block_stride=args.block_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_dl, val_dl, test_dl = dl.get_dataloaders()
    logging.info(f"Train size: {len(train_dl.dataset)}")
    logging.info(f"Val size:   {len(val_dl.dataset)}")
    logging.info(f"Test size:  {len(test_dl.dataset)}")

    # define model
    encoder = CPCEncoder(16, 256)
    ar = CPCAR(256, 256, 2)
    model = SupervisedModel(encoder, ar, args)

    # define loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_cpc, weight_decay=args.weight_decay_cpc
    )
    lr_reduce = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True
    )

    # train
    metrics: Dict[str, List] = {"train": [], "val": [], "test": []}
    for epoch in range(2, 1 + args.epochs_cpc):
        metrics["train"].append(train_one_epoch(model, train_dl, criterion, optimizer, epoch, args))
        metrics["val"].append(validate(model, val_dl, criterion, epoch, args))

        lr_reduce.step(metrics["val"][-1]["loss"])

        # save model
        save_checkpoint(model, epoch, args, metrics["val"])

    # test
    metrics["test"].append(test(model, test_dl, criterion, epoch, args))

    end = time.time()
    logging.info(f"Total time: {end-start:.2f}s")

    cleanup(args)


if __name__ == "__main__":
    args = setup()
    try:
        main(args)
    except (Exception, KeyboardInterrupt) as e:
        cleanup(args)
        raise e
