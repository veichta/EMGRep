"""Implementation of the training loop for CPC model."""

import datetime
import logging
import time
from argparse import Namespace
from typing import Tuple

from torch.utils.data import DataLoader

# from emgrep.criterion import CPCCriterion
from emgrep.models.cpc_model import CPCAR, CPCEncoder, CPCModel


def train_cpc(args: Namespace) -> Tuple[CPCModel, dict[str, DataLoader]]:
    """Train the model.

    Args:
        args (Namespace): Command line arguments.

    Returns:
        Tuple[CPCModel, dict[str, DataLoader]]: Trained model and dataloaders.
    """
    logging.info("Training the model...")
    start = time.time()

    # TODO: Initialize

    encoder = CPCEncoder(in_channels=16, hidden_dim=args.encoder_dim)
    ar = CPCAR(dimEncoded=args.encoder_dim, dimOutput=args.ar_dim, numLayers=args.ar_layers)
    cpc_model = CPCModel(encoder=encoder, ar=ar)
    # criterion = CPCCriterion(k=args.cpc_k)
    dataloaders = {
        "train": None,
        "val": None,
        "test": None,
    }

    # TODO: Train model
    # metrics = {"train": {}, "val": {}, "test": {}}
    # for epoch in range(args.epochs_cpc):
    #     train_one_epoch(...)
    #     validate(...)
    #     save_checkpoint(...)

    # test(...)

    end = time.time()
    elapsed = datetime.timedelta(seconds=end - start)
    logging.info(f"Training time: {elapsed}")

    # Log metrics

    return cpc_model, dataloaders
