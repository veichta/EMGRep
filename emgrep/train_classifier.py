"""Implementation of the training loop for classifier."""

import datetime
import logging
import time
from argparse import Namespace

from torch.utils.data import DataLoader, Dataset


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
