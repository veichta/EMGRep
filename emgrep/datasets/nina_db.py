"""Datasets for the NINA-PRO dataset."""

from typing import Any

from torch.utils.data import Dataset


class SingleFileEMGDataLoader(Dataset):
    """Dataset for single file of NINA-PRO."""

    def __init__(
        self,
        mat_file: dict[str, Any],
        sec_len: int,
        target_transform: Any = None,
        window_size: int = 100,
        stride: int = 1,
        rms_stride: int = 1,
        rms_padding: int = 0,
    ):
        """Initialize the dataset.

        Args:
            mat_file (dict[str, Any]): GMP data file.
            sec_len (int): Window Size.
            target_transform (Any, optional): Transform to apply to the target. Defaults to None.
            window_size (int, optional): Block size in the window. Defaults to 100.
            stride (int, optional): Stride for the window. Defaults to 1.
            rms_stride (int, optional): Stride for the RMS. Defaults to 1.
            rms_padding (int, optional): Padding for the RMS. Defaults to 0.
        """
        super().__init__()

        self.mat_file = mat_file
        self.sec_len = sec_len
        self.target_transform = target_transform
        self.window_size = window_size
        self.stride = stride
        self.rms_stride = rms_stride
        self.rms_padding = rms_padding

        self.emg = self.mat_file["emg"]
        self.label = self.mat_file["restimulus"]

    def __len__(self):
        """Get the length of the dataset."""
        pass

    def __getitem__(self, idx: int):
        """Get the item at the given index.

        Args:
            idx (int): Index of the item.
        """
        pass
