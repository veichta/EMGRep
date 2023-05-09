"""Dataset for the EMGRep project."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from emgrep.utils.preprocessing import hilbert_envelope, rms_preprocess, savgol_preprocess


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
        normalize: bool = True,
        preprocessing: str = "rms",
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
            normalize (bool, optional): Whether to standardize features to zero mean
                and unit variance (as last preprocessing step). Defaults to True.
            preprocessing (str, optional): What type of preprocessing to apply.
                Should be one of [None, "rms", "savgol", "hilbert"],
                defaults to RMS amplitude smoothing
        """
        super().__init__()

        self.positive_mode = positive_mode
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.block_len = block_len
        self.block_stride = block_stride
        self.normalize = normalize
        self.preprocessing = preprocessing

        assert self.positive_mode in {
            "none",
            "subject",
            "session",
            "label",
        }, "Positive mode must be 'none', 'subject', 'session' or 'label'."

        self.emg, self.stimulus, self.info = self._load_data(mat_files)

        self.rng = np.random.default_rng(seed=42)

    def _load_data(
        self, mat_files: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Creates sequences from the data.

        Args:
            mat_files (List[Dict[str, Any]]): List containing the mat files.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: EMG, stimulus and info.
        """
        emg = []
        stimulus = []
        info = []

        for mat_file in mat_files:
            signal = mat_file["emg"]
            label = mat_file["restimulus"]

            if self.preprocessing == "rms":
                signal = rms_preprocess(signal)
            elif self.preprocessing == "savgol":
                signal = savgol_preprocess(signal)
            elif self.preprocessing == "hilbert":
                signal = hilbert_envelope(signal)

            if self.normalize:
                signal -= signal.mean(axis=0)[None, :]
                signal /= 1e-8 + signal.std(axis=0)[None, :]

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
                            int(stimulus[-1][self.seq_len // 2]),
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

    # TODO: needs to be corrected
    def _sample_positive_seq(self, info: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples a positive sequence based on the positive mode.

        Args:
            info (np.ndarray): Information of the sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: EMG, stimulus, and info of positive sample.

        Notes:
            The positive mode controls a subset of data the positive samples are drawn from. It can
            be one of the following:
                - subject: Positive samples must come from the same subject.
                - session: Positive samples must come from the same session.
                - label: No additional constraint.
        """
        assert self.positive_mode in {
            "subject",
            "session",
            "label",
        }, "Positive mode must be 'subject', 'session' or 'label'."

        if self.positive_mode == "subject":
            positive_mode_condition = np.all(self.info[:, [0, 3]] == info[[0, 3]], axis=1)

        if self.positive_mode == "session":
            positive_mode_condition = np.all(self.info == info, axis=1)

        if self.positive_mode == "label":
            positive_mode_condition = self.info[:, -1] == info[-1]

        positive_indices = positive_mode_condition.nonzero()[0]
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
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: EMG, stimulus and info. Note that EMG
            is of shape (batch_size, 2 (or 1), num_blocks, block_size, num_sensors) and stimulus is
            of shape (batch_size, 2 (or 1), num_blocks, block_size, 1). Info is of shape
            (batch_size, 2 (or 1), 4).
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

        return (
            torch.from_numpy(emg).float(),
            torch.from_numpy(stimulus).float(),
            torch.from_numpy(info),
        )
