"""Utility functions and classes."""

import argparse
import datetime
import logging
import os

from emgrep.utils.args import parse_args


def setup() -> argparse.Namespace:
    """Setup arguments and logging.

    Returns:
        argparse.Namespace: Parsed arguments from command line.
    """
    args = parse_args()

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
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(args.log_dir, f"{date}_log.txt")
    else:  # log to stdout
        log_file = None

    logging.basicConfig(
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
        filename=log_file,
    )
