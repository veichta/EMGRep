"""Dataset for the EMGRep project."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class EMGRepDataset(Dataset):
    """Dataset for the EMGRep project."""

    def __init__(
        self,
        mat_files: List[Dict[str, Any]],
        positive_mode: str = "none",
        seq_len: int = 3000,
        seq_stride: int = 3000,
        block_len: int = 300,
        block_stride: int = 300,
    ) -> None:
        """Initialize the dataset.

        Args:
            mat_files (List[Dict[str, Any]]): List containing the mat files.
            positive_mode (str, optional): Whether to use self or subject as positive class.
                Defaults to "none". Other options are: "session", "subject", "label".
            seq_len (int, optional): Length of the sequence. Defaults to 3000.
            seq_stride (int, optional): Stride of the sequence. Defaults to 3000.
            block_len (int, optional): Length of the block in sequence. Defaults to 300.
            block_stride (int, optional): Stride of the block in sequence. Defaults to 300.
        """
        super().__init__()

        self.mat_files = mat_files
        self.positive_mode = positive_mode
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.block_len = block_len
        self.block_stride = block_stride

        assert self.positive_mode in {
            "none",
            "subject",
            "session",
            "label",
        }, "Positive mode must be 'none', 'subject', 'session' or 'label'."

        self.emg, self.stimulus, self.info = self._load_data()

        self.rng = np.random.default_rng(seed=42)

    def _load_data(self):
        """Creates sequences from the data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: EMG, stimulus and info.
        """
        emg = []
        stimulus = []
        info = []

        for mat_file in self.mat_files:
            signal = mat_file["emg"]
            label = mat_file["restimulus"]

            idx = 0
            while idx + self.seq_len <= signal.shape[0]:
                emg.append(signal[idx : idx + self.seq_len])
                stimulus.append(label[idx : idx + self.seq_len])
                info.append(
                    np.array(
                        [
                            mat_file["subj"][0, 0],
                            mat_file["daytesting"][0, 0],
                            mat_file["time"][0, 0],
                            int(stimulus[-1][-1]),
                        ]
                    )
                )
                idx += self.seq_stride

        emg = np.stack(emg)
        stimulus = np.stack(stimulus)
        info = np.stack(info)

        return emg, stimulus, info

    def _seq_to_blocks(self, signal) -> np.ndarray:
        """Converts a sequence to blocks.

        Args:
            signal (np.ndarray): input signal.

        Returns:
            np.ndarray: blocks.
        """
        blocks = []

        idx = 0
        while idx + self.block_len <= signal.shape[0]:
            blocks.append(signal[idx : idx + self.block_len])
            idx += self.block_stride

        return np.stack(blocks)

    def _sample_positive_seq(self, info: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples a positive sequence based on the positive mode.

        Args:
            info (np.ndarray): Information of the sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: EMG, stimulus, and info of positive sample.
        """
        assert self.positive_mode != "none", "Positive mode must not be 'none'."

        stimulus_condition = self.stimulus[:, -1] == info[-1]
        if self.positive_mode == "subject":
            positive_mode_condition = self.info[:, 0] == info[0]

        if self.positive_mode == "session":
            positive_mode_condition = np.all(self.info == info, axis=1)

        if self.positive_mode == "label":
            positive_mode_condition = self.stimulus[:, -1] == info[-1]

        positive_indices = np.logical_and(stimulus_condition, positive_mode_condition).nonzero()[0]
        positive_idx = self.rng.choice(positive_indices)

        return self.emg[positive_idx], self.stimulus[positive_idx], self.info[positive_idx]

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.emg.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: EMG, stimulus and info.
        """
        emg_blocks = self._seq_to_blocks(self.emg[idx])
        stimulus_blocks = self._seq_to_blocks(self.stimulus[idx])
        info = self.info[idx]

        if self.positive_mode != "none":
            positive_emg, positive_stimulus, positive_info = self._sample_positive_seq(info)
            positive_emg_blocks = self._seq_to_blocks(positive_emg)
            positive_stimulus_blocks = self._seq_to_blocks(positive_stimulus)

            emg = np.stack([emg_blocks, positive_emg_blocks])
            stimulus = np.stack([stimulus_blocks, positive_stimulus_blocks])
            info = np.stack([info, positive_info])
        else:
            emg = np.expand_dims(emg_blocks, 0)
            stimulus = np.expand_dims(stimulus_blocks, 0)
            info = np.expand_dims(info, 0)

        return torch.from_numpy(emg), torch.from_numpy(stimulus), torch.from_numpy(info)
