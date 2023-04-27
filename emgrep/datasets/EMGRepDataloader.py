"""Dataloader for the EMGRep project."""

import logging
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader
from tqdm import tqdm

from emgrep.datasets.EMGRepDataset import EMGRepDataset
from emgrep.utils.io import get_recording


class EMGRepDataloader:
    """Dataloader class."""

    def __init__(
        self,
        data_path: Path,
        data_selection: List[Tuple[int, int, int]] = [],
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
            data_selection (List[Tuple[int, int, int]], optional): Data selection.
                Defaults to [] meaning all. Given as a list of tuples (subject, day, time).
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
        self.data_selection = data_selection
        if not data_selection:
            self.data_selection = [
                (subject, day, time)
                for subject in [1, 2, 3, 7, 8, 9, 10]
                for day in [1, 2, 3, 4, 5]
                for time in [1, 2]
            ]
        self.positive_mode = positive_mode
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.block_len = block_len
        self.block_stride = block_stride
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = self._create_dataset()

    def _create_dataset(self) -> EMGRepDataset:
        """Create the dataset."""
        logging.info("Loading data...")
        mat_files = [
            get_recording(subject=subject, day=day, time=time, data_path=str(self.data_path))
            for subject, day, time in tqdm(self.data_selection)
        ]
        return EMGRepDataset(
            mat_files=mat_files,
            positive_mode=self.positive_mode,
            seq_len=self.seq_len,
            seq_stride=self.seq_stride,
            block_len=self.block_len,
            block_stride=self.block_stride,
        )

    def get_dataloader(self) -> DataLoader:
        """Get the dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
