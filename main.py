"""This project aims to learn emg data representations for hand movement recognition."""

import datetime
import logging
import time
from argparse import Namespace

from emgrep.datasets.EMGRepDataloader import get_dataloader
from emgrep.datasets.RepresentationsDataset import RepresentationDataset
from emgrep.train_classifier import train_classifier
from emgrep.train_cpc import train_cpc
from emgrep.utils.utils import cleanup, setup
from emgrep.visualizations import visualize_embeddings


def main(args: Namespace):
    """Main function."""
    start = time.time()

    # TODO: Load data
    dataloaders = get_dataloader(args)

    # TODO: Train model
    model = train_cpc(dataloaders, args)

    # TODO: Extract representations
    representations = {
        "train": RepresentationDataset(model=model, dataloader=dataloaders["train"], args=args),
        "val": RepresentationDataset(model=model, dataloader=dataloaders["val"], args=args),
        "test": RepresentationDataset(model=model, dataloader=dataloaders["test"], args=args),
    }

    # TODO: Evaluate representations
    train_classifier(representations, dataloaders, args)

    # TODO: Visualize representations
    visualize_embeddings(representations, dataloaders, args)

    end = time.time()
    elapsed = datetime.timedelta(seconds=end - start)
    logging.info(f"Elapsed time: {elapsed}")

    cleanup(args)


if __name__ == "__main__":
    args = setup()
    try:
        main(args)
    except (Exception, KeyboardInterrupt) as e:
        cleanup(args)
        raise e
