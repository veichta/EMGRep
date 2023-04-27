"""Script for parsing command line arguments."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="emgrep: Representation learning framework of emp data for hand \
        movement recognition"
    )
    # DATA
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Path to data directory.",
    )

    # MODEL

    # TRAINING CPC
    parser.add_argument(
        "--epochs_cpc",
        type=int,
        default=10,
        help="Number of epochs for training CPC.",
    )
    parser.add_argument(
        "--encoder_dim",
        type=int,
        default=258,
        help="Dimension of encoder output.",
    )
    parser.add_argument(
        "--ar_dim",
        type=int,
        default=258,
        help="Dimension of autoregressive model output.",
    )
    parser.add_argument(
        "--ar_layers",
        type=int,
        default=2,
        help="Number of layers in autoregressive model.",
    )
    parser.add_argument(
        "--cpc_k",
        type=int,
        default=5,
        help="Number of steps for contrastive prediction.",
    )

    # TRAINING CLASSIFIER
    parser.add_argument(
        "--epochs_classifier",
        type=int,
        default=10,
        help="Number of epochs for training classifier.",
    )

    # LOGGING
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Path to log directory.",
    )
    return parser.parse_args()
