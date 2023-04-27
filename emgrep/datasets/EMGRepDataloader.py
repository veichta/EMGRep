"""Dataloader for the EMGRep project."""

from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader

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
        if mode == "train":
            data = self.train_data
        elif mode == "val":
            data = self.val_data
        elif mode == "test":
            data = self.test_data
        mat_files = []
        for subject, day, time in data:
            mat_files.append(
                get_recording(
                    subject=subject,
                    day=day,
                    time=time,
                    data_path=str(self.data_path),
                )
            )

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
