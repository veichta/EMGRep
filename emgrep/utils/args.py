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
    # SETUP
    parser.add_argument(
        "--data",
        type=str,
        default="data/01_raw",
        help="Path to data directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs",
        help="Where to save output (e.g. plots).",
    )

    # DATA LOADER
    parser.add_argument(
        "--n_subjects",
        type=int,
        default=10,
        help="Number of subjects to use.",
    )
    parser.add_argument(
        "--n_days",
        type=int,
        default=5,
        help="Number of days to use.",
    )
    parser.add_argument(
        "--n_times",
        type=int,
        default=2,
        help="Number of sessions to use.",
    )
    parser.add_argument(
        "--positive_mode",
        type=str,
        default="none",
        choices=["none", "session", "subject", "label"],
        help="Whether to use self or subject as positive class.",
    )
    parser.add_argument(
        "--val_idx",
        type=int,
        default=1,
        help="Index of subject or day to use for validation.",
    )
    parser.add_argument(
        "--test_idx",
        type=int,
        default=2,
        help="Index of subject or day to use for testing.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=3_000,
        help="Length of sequence.",
    )
    parser.add_argument(
        "--seq_stride",
        type=int,
        default=3_000,
        help="Stride of sequence.",
    )
    parser.add_argument(
        "--block_len",
        type=int,
        default=300,
        help="Length of block in sequence.",
    )
    parser.add_argument(
        "--block_stride",
        type=int,
        default=300,
        help="Stride of block in sequence.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for dataloader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader.",
    )

    # MODEL
    parser.add_argument(
        "--encoder_dim",
        type=int,
        default=256,
        help="Dimension of encoder output.",
    )
    parser.add_argument(
        "--ar_dim",
        type=int,
        default=256,
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

    # TRAINING CPC MODEL
    parser.add_argument(
        "--epochs_cpc",
        type=int,
        default=100,
        help="Number of epochs for training CPC.",
    )
    parser.add_argument(
        "--lr_cpc",
        type=float,
        default=1e-2,
        help="Learning rate for training CPC.",
    )
    parser.add_argument(
        "--weight_decay_cpc",
        type=float,
        default=0.0,
        help="Weight decay for training CPC.",
    )

    # TRAINING CLASSIFIER
    parser.add_argument(
        "--epochs_classifier",
        type=int,
        default=50,
        help="Number of epochs for training classifier.",
    )
    parser.add_argument(
        "--batch_size_classifier",
        type=int,
        default=64,
        help="Batch Size for training classifier.",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=12,
        help="Number of classes classifier.",
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
        default="logs",
        help="Path to log directory.",
    )
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        help="Log to file.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to wandb.",
    )

    return parser.parse_args()
