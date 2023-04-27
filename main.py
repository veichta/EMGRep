"""This project aims to learn emg data representations for hand movement recognition."""

import datetime
import logging
import time

from emgrep.datasets.RepresentationsDataset import RepresentationDataset
from emgrep.train_classifier import train_classifier
from emgrep.train_cpc import train_cpc
from emgrep.utils.utils import setup
from emgrep.visualizations import visualize_embeddings


def main():
    """Main function."""
    args = setup()
    start = time.time()

    # TODO: Train model
    model, dataloaders = train_cpc(args)

    # TODO: Extract representations
    representations = {
        "train": RepresentationDataset(model=model, dataloader=dataloaders["train"]),
        "val": RepresentationDataset(model=model, dataloader=dataloaders["val"]),
        "test": RepresentationDataset(model=model, dataloader=dataloaders["test"]),
    }

    # TODO: Evaluate representations
    train_classifier(representations, dataloaders, args)

    # TODO: Visualize representations
    visualize_embeddings(representations, dataloaders, args)

    end = time.time()
    elapsed = datetime.timedelta(seconds=end - start)
    logging.info(f"Elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
