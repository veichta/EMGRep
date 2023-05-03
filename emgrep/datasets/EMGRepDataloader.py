"""Dataloader for the EMGRep project."""

import logging
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader
from tqdm import tqdm

from emgrep.datasets.EMGRepDataset import EMGRepDataset
from emgrep.utils.io import get_recording


class EMGRepDataloader:
    """Dataloader class."""

    def __init__(
        self,
        data_path: Path,
        train_data: List[Tuple[int, int, int]],
        val_data: List[Tuple[int, int, int]] = [],
        test_data: List[Tuple[int, int, int]] = [],
        positive_mode: str = "none",
        seq_len: int = 3000,
        seq_stride: int = 3000,
        block_len: int = 300,
        block_stride: int = 300,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        """Initialize the dataloader.

        Args:
            data_path (Path): Path to the data.
            train_data (List[Tuple[int, int, int]]): Data selected for training.
                Given as a list of tuples (subject, day, time).
            val_data (List[Tuple[int, int, int]], optional): Data selected for validation.
                Given as a list of tuples (subject, day, time). Defaults to []. If empty,
                no validation dataloader can be created.
            test_data (List[Tuple[int, int, int]], optional): Data selected for testing.
                Given as a list of tuples (subject, day, time). Defaults to []. If empty,
                no testing dataloader can be created.
            positive_mode (str, optional): Whether to use self or subject as positive class.
                Defaults to "none". Other options are: "session", "subject", "label".
            seq_len (int, optional): Length of the sequence. Defaults to 3000.
            seq_stride (int, optional): Stride of the sequence. Defaults to 3000.
            block_len (int, optional): Length of the block in sequence. Defaults to 300.
            block_stride (int, optional): Stride of the block in sequence. Defaults to 300.
            batch_size (int, optional): Batch size for the dataloader. Defaults to 1.
            num_workers (int, optional): Number of workers for the dataloader. Defaults to 0.
        """
        super().__init__()
        self.data_path = data_path
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.positive_mode = positive_mode
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.block_len = block_len
        self.block_stride = block_stride
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_dataset(self, mode="train") -> EMGRepDataset:
        """Create the dataset.

        Args:
            mode (str, optional): Mode of the dataset. Defaults to "train".

        Returns:
            EMGRepDataset: Dataset.
        """
        if mode == "test":
            data = self.test_data

        elif mode == "train":
            data = self.train_data
        elif mode == "val":
            data = self.val_data

        logging.info(f"Loading {mode} dataset...")
        mat_files = [
            get_recording(
                subject=subject,
                day=day,
                time=time,
                data_path=str(self.data_path),
            )
            for subject, day, time in tqdm(data)
        ]
        return EMGRepDataset(
            mat_files=mat_files,
            positive_mode=self.positive_mode,
            seq_len=self.seq_len,
            seq_stride=self.seq_stride,
            block_len=self.block_len,
            block_stride=self.block_stride,
        )

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get the dataloader.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation and test dataloader.
        """
        train_dataset = self._create_dataset(mode="train")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if self.val_data:
            val_dataset = self._create_dataset(mode="val")
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        test_loader = None
        if self.test_data:
            test_dataset = self._create_dataset(mode="test")
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        return train_loader, val_loader, test_loader


def get_dataloader(args: Namespace) -> Dict[str, DataLoader]:
    """Get the dataloaders.

    Args:
        args (Namespace): Command line arguments.

    Returns:
        Dict[str, DataLoader]: Train, val, and test dataloaders.
    """
    train_split, val_split, test_split = get_split(args)

    dl = EMGRepDataloader(
        data_path=args.data,
        train_data=train_split,
        val_data=val_split,
        test_data=test_split,
        positive_mode=args.positive_mode,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        block_len=args.block_len,
        block_stride=args.block_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_dl, val_dl, test_dl = dl.get_dataloaders()

    logging.info(f"Train samples: {len(train_dl.dataset)}")
    logging.info(f"Val   samples: {len(val_dl.dataset)}")
    logging.info(f"Test  samples: {len(test_dl.dataset)}")

    return {
        "train": train_dl,
        "val": val_dl,
        "test": test_dl,
    }


def get_split(
    args: Namespace,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Get the train, val, and test splits.

    Args:
        args (Namespace): Command line arguments.

    Raises:
        ValueError: If the positive mode is not recognized.

    Returns:
        List[Tuple[int, int, int]]: Train, val, and test splits.
    """
    subject_range = range(1, args.n_subjects + 1)
    day_range = range(1, args.n_days + 1)
    time_range = range(1, args.n_times + 1)

    if args.split_mode == "day":
        assert args.val_idx in day_range, f"Invalid val index: {args.val_idx}"
        assert args.test_idx in day_range, f"Invalid test index: {args.test_idx}"
        train_split = [
            (subject, day, time)
            for subject in subject_range
            for day in day_range
            for time in time_range
            if day not in [args.val_idx, args.test_idx]
        ]
        val_split = [
            (subject, day, time)
            for subject in subject_range
            for day in [args.val_idx]
            for time in time_range
        ]
        test_split = [
            (subject, day, time)
            for subject in subject_range
            for day in [args.test_idx]
            for time in time_range
        ]
    elif args.split_mode == "subject":
        assert args.val_idx in subject_range, f"Invalid val index: {args.val_idx}"
        assert args.test_idx in subject_range, f"Invalid test index: {args.test_idx}"
        train_split = [
            (subject, day, time)
            for subject in subject_range
            for day in day_range
            for time in time_range
            if subject not in [args.val_idx, args.test_idx]
        ]
        val_split = [
            (subject, day, time)
            for subject in [args.val_idx]
            for day in day_range
            for time in time_range
        ]
        test_split = [
            (subject, day, time)
            for subject in [args.test_idx]
            for day in day_range
            for time in time_range
        ]
    else:
        raise ValueError(f"Invalid positive mode: {args.positive_mode}")

    return train_split, val_split, test_split
