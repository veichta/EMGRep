"""Datasets for the NINA-PRO dataset."""

import logging
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class SingleFileEMGDataLoader(Dataset):
    """Dataset for single file of NINA-PRO."""

    def __init__(
        self,
        mat_file: Dict[str, Any],
        positives: str = "self",
        sec_len: int = 3000,
        block_len: int = 300,
        block_stride: int = 300,
        target_transform: Any = None,
        rms_window: int = 30,
        rms_stride: int = 1,
        rms_padding: int = 0,
    ):
        """Initialize the dataset.

        Args:
            mat_file (Dict[str, Any]): Dictionary containing the data.
            positives (str, optional): Whether to use self or subject as positive class. Defaults to
            "self".
            sec_len (int, optional): Length of the sequence. Defaults to 3000.
            block_len (int, optional): Length of the block in sequence. Defaults to 300.
            block_stride (int, optional): Stride of the block in sequence. Defaults to 300.
            target_transform (Any, optional): Target transform. Defaults to None.
            rms_window (int, optional): Window for the RMS. Defaults to 30.
            rms_stride (int, optional): Stride for the RMS. Defaults to 1.
            rms_padding (int, optional): Padding for the RMS. Defaults to 0.
        """
        super().__init__()

        self.mat_file = mat_file
        self.positives = positives
        self.sec_len = sec_len
        self.block_len = block_len
        self.block_stride = block_stride
        self.target_transform = target_transform
        self.rms_window = rms_window
        self.rms_stride = rms_stride
        self.rms_padding = rms_padding

        assert self.positives in {"self", "subject"}, "Positives must be 'self' or 'subject'."

        self.signal = self.mat_file["emg"]
        self.stimulus = self.mat_file["restimulus"]
        self.classes = np.unique(self.stimulus)

        self.data = self.get_sequences()
        self.label_samples = self.get_label_to_seq()

    def get_sequences(self) -> List[tuple[Any, Any, Any]]:
        """Get the sequences.

        Returns:
            List[tuple[Any, Any]]: List of sequences.
        """
        info = {
            "subject": self.mat_file["subj"][0, 0],
            "day": self.mat_file["daytesting"][0, 0],
            "time": self.mat_file["time"][0, 0],
        }
        logging.info(
            f"Getting sequences for {info['subject']} at day {info['day']} at time {info['time']}."
        )
        return [
            (
                self.signal[i : i + self.sec_len + 1],
                self.stimulus[i : i + self.sec_len + 1, 0],
                info,
            )
            for i in range(0, self.signal.shape[0] - self.sec_len, self.sec_len)
        ]

    def split_into_block(self, signal: Any, stimulus: Any):
        """Split the signal into blocks.

        Args:
            signal (Any): Signal.
            stimulus (Any): Stimulus.


        """
        sig_blocks = [
            signal[i : i + self.block_len]
            for i in range(0, signal.shape[0] - self.block_len, self.block_stride)
        ]
        stim_blocks = [
            stimulus[i : i + self.block_len]
            for i in range(0, stimulus.shape[0] - self.block_len, self.block_stride)
        ]
        return np.array(sig_blocks), np.array(stim_blocks)

    def get_label_to_seq(self) -> Dict[int, List[int]]:
        """Get the label to sequence mapping.

        Args:
            seq (Any): Sequence.
            labels (Any): Labels.

        Returns:
            Dict[int, List[Any]]: Label to sequence mapping.
        """
        label_to_seq: Dict[int, List[int]] = {c: [] for c in self.classes}
        for i, d in enumerate(self.data):
            _, labels, _ = d
            label = labels[-1]
            label_to_seq[label].append(i)

        return label_to_seq

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[tuple[Any, Any], tuple[Any, Any], Any]:
        """Get the item at the given index.

        Args:
            idx (int): Index of the item.
        """
        seq, labels, info = self.data[idx]

        pos_sequences = self.label_samples[labels[-1]]

        if self.positives == "self":
            pos_idx = idx
        elif self.positives == "subject":
            pos_idx = np.random.choice(self.label_samples[labels[-1]])

        pos_seq, pos_labels, _ = self.data[pos_sequences[pos_idx]]

        seq_blocks, label_blocks = self.split_into_block(seq, labels)
        pos_seq_blocks, pos_label_blocks = self.split_into_block(pos_seq, pos_labels)

        return (
            (torch.tensor(seq_blocks), torch.tensor(label_blocks)),
            (torch.tensor(pos_seq_blocks), torch.tensor(pos_label_blocks)),
            info,
        )
