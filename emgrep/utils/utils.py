"""Utility functions and classes."""

import argparse
import datetime
import logging
import os

import wandb

from emgrep.utils.args import parse_args


def setup() -> argparse.Namespace:
    """Setup arguments and logging.

    Returns:
        argparse.Namespace: Parsed arguments from command line.
    """
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(
        args.log_dir, f"{args.positive_mode}_{args.val_idx}_{args.test_idx}", timestamp
    )
    args.log_dir = log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    setup_logging(args)

    logging.debug("Command line arguments:")
    for arg, val in vars(args).items():
        logging.debug(f"\t{arg}: {val}")

    return args


def setup_logging(args: argparse.Namespace):
    """Setup logging.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    log_file = os.path.join(args.log_dir, "log.txt") if args.log_to_file else None
    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
        filename=log_file,
    )

    # Suppress logging from other modules
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if args.wandb:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            entity="sjohn-eth",
            project="emgrep",
            name=f"{args.positive_mode}_{args.val_idx}_{args.test_idx}_{timestamp}",
            config=vars(args),
            dir=args.log_dir,
        )


def cleanup(args: argparse.Namespace):
    """Cleanup logging.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    if args.wandb:
        wandb.finish()

    if len(os.listdir(args.log_dir)) == 0:
        os.rmdir(args.log_dir)
