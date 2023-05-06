"""This project aims to learn emg data representations for hand movement recognition."""

import datetime
import logging
import time
from argparse import Namespace

from emgrep.datasets.EMGRepDataloader import get_dataloader
from emgrep.datasets.RepresentationsDataset import RepresentationDataset

# from emgrep.train_classifier import train_classifier
from emgrep.train_classifier import train_classifier
from emgrep.train_cpc import train_cpc
from emgrep.utils.utils import cleanup, setup

# from emgrep.visualizations import visualize_embeddings


def main(args: Namespace):
    """Main function."""
    start = time.time()

    # TODO: Load data
    cpc_dataloaders = get_dataloader(args)

    # TODO: Train model
    model = train_cpc(cpc_dataloaders, args)

    del cpc_dataloaders

    # TODO: Extract representations
    rep_dataloader = get_dataloader(args, block_stride=True)
    representations = {
        "train": RepresentationDataset(model=model, dataloader=rep_dataloader["train"], args=args),
        "val": RepresentationDataset(model=model, dataloader=rep_dataloader["val"], args=args),
        "test": RepresentationDataset(model=model, dataloader=rep_dataloader["test"], args=args),
    }
    # representations = {}
    # for phase in ["train", "val", "test"]:
    #     representations[phase] = RepresentationDataset(
    #         model=model, dataloader=rep_dataloader[phase], args=args
    #     )
    #     del representations[phase]
    del rep_dataloader

    # TODO: Evaluate representations
    train_classifier(representations, pred_block=-1, args=args)
    # train_classifier(representations, pred_block=-2, args=args)
    # train_classifier(representations, pred_block=-3, args=args)
    # train_classifier(representations, pred_block=-4, args=args)
    # train_classifier(representations, pred_block=-5, args=args)

    # TODO: Visualize representations
    # visualize_embeddings(representations, rep_dataloader, args)

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
