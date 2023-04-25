"""Define the dataloader class for the NINA-DB dataset."""

from typing import Tuple

import torch

from emgrep.datasets.nina_db import SingleFileEMGDataLoader
from emgrep.utils.io import get_recording

label_values = [0, 1, 3, 4, 6, 9, 10, 11]
n_classes = len(label_values)
label_dict = {label_values[i]: i for i in range(n_classes)}


def get_end_label(labels: list) -> int:
    """Get the label of the last sample in the block.

    Args:
        labels (list): List of labels.

    Returns:
        int: Label of the last sample in the block.
    """
    return label_dict[labels[-1]]


def get_mid_label(labels: list) -> int:
    """Get the label of the middle sample in the block.

    Args:
        labels (list): List of labels.

    Returns:
        int: Label of the middle sample in the block.
    """
    return label_dict[labels[len(labels) // 2]]


class Dataloader:
    """Dataloader class.

    Call either intra_subject_data(subject) or inter_subject_data() to get the full respective
    datasets.
    Call get_dataloaders(data) to get the dataloader.
    """

    def __init__(
        self,
        sec_len: int,
        block_len: int,
        block_stride: int,
        positives: str = "self",
        batch_size: int = 1,
        num_workers: int = 4,
        days: list = list(range(1, 6)),
        times: Tuple[int, int] = (1, 2),
        rms_window: int = 1,
        rms_stride: int = 1,
        rms_padding: int = 0,
    ):
        """Initialize the dataloader.

        Args:
            sec_len (int): Length of each sequence.
            block_len (int): Length of each block in the sequence.
            block_stride (int): Stride of the blocks in the sequence.
            positives (str, optional): Whether to use self or subject positives. Defaults to "self".
            batch_size (int, optional): Batch size for the dataloader. Defaults to 1.
            num_workers (int, optional): Number of workers for the dataloader. Defaults to 4.
            days (list, optional): Days to load. Defaults to list(range(1, 6)).
            times (Tuple[int, int], optional): Times to load. Defaults to (1, 2).
            rms_window (int, optional): Window size for the RMS. Defaults to 1.
            rms_stride (int, optional): Stride for the RMS. Defaults to 1.
            rms_padding (int, optional): Padding for the RMS. Defaults to 0.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sec_len = sec_len
        self.block_len = block_len
        self.block_stride = block_stride
        self.positives = positives
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.days = days
        self.times = times
        self.rms_window = rms_window
        self.rms_stride = rms_stride
        self.rms_padding = rms_padding

    def get_dataloaders(self, data: torch.utils.data.ConcatDataset) -> tuple:
        """Get the train, validation and test dataloaders.

        Args:
            data (torch.utils.data.ConcatDataset): Concatenated dataset of all recordings.

        Returns:
            tuple: Tuple containing the train, validation and test dataloaders.
        """
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))
        test_size = len(data) - train_size - val_size
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(
            data, [train_size, val_size, test_size]
        )

        train_loader = self._get_loader(self.train_data)
        val_loader = self._get_loader(self.val_data)
        test_loader = self._get_loader(self.test_data)
        return train_loader, val_loader, test_loader

    def _get_loader(self, dataset: torch.utils.data.ConcatDataset) -> torch.utils.data.DataLoader:
        """Get the dataloader.

        Args:
            dataset (torch.utils.data.ConcatDataset): Concatenated dataset of all recordings.

        Returns:
            torch.utils.data.DataLoader: Dataloader.
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def intra_subject_data(self, data_path: str, subject: int) -> torch.utils.data.ConcatDataset:
        """Load the data from a single subject.

        Args:
            data_path (str): Path to the dataset.
            subject (int): Id of the subject (1-10).

        Returns:
            torch.utils.data.ConcatDataset: Concatenated dataset of all recordings of the subject.
        """
        datasets = []
        for day in self.days:
            for time in self.times:
                f = get_recording(subject=subject, day=day, time=time, data_path=data_path)
                datasets.append(
                    # TODO: update this if needed
                    SingleFileEMGDataLoader(
                        f,
                        sec_len=self.sec_len,
                        positives=self.positives,
                        block_len=self.block_len,
                        block_stride=self.block_stride,
                        target_transform=get_end_label,
                        rms_window=self.rms_window,
                        rms_stride=self.rms_stride,
                        rms_padding=self.rms_padding,
                    )
                )
        return torch.utils.data.ConcatDataset(datasets)

    def inter_subject_data(self, data_path: str, subjects: list) -> torch.utils.data.ConcatDataset:
        """Load the data from multiple subjects.

        Args:
            data_path (str): Path to the dataset.
            subjects (list, optional): List of subjects to load.

        Returns:
            torch.utils.data.ConcatDataset: Concatenated dataset of all subjects.
        """
        datasets = [self.intra_subject_data(data_path=data_path, subject=s) for s in subjects]
        return torch.utils.data.ConcatDataset(datasets)
