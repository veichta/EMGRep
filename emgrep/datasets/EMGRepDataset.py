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


class EMGRepDatasetEfficient(Dataset):
    """Dataset for the EMGRep project."""

    def __init__(
        self,
        mat_files: List[Dict[str, Any]],
        positive_mode: str = "none",
        seq_len: int = 3000,
        seq_stride: int = 3000,
        block_len: int = 300,
        block_stride: int = 300,
        normalize: bool = False,
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

        self.seq_map = self._get_seq_map()
        self.pos_map = self._get_pos_map()

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
        labels = []
        infos: List[Dict[str, Any]] = []
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

            info = {
                "subj": mat_file["subj"][0, 0],
                "daytesting": mat_file["daytesting"][0, 0],
                "time": mat_file["time"][0, 0],
                "label": int(label[self.seq_len // 2]),
            }

            emg.append(signal)
            labels.append(label)
            infos.extend(info for _ in range(signal.shape[0]))

        emg_arr = np.concatenate(emg)
        labels_arr = np.concatenate(labels)
        infos_arr = np.array(infos)

        assert emg_arr.shape[0] == labels_arr.shape[0] == infos_arr.shape[0], "Shapes do not match."

        return emg_arr, labels_arr, infos_arr

    def _get_seq_map(self) -> np.ndarray:
        """Get map for sequences.

        Returns:
            np.ndarray: Sequence map.
        """
        seq_map = []

        idx = 0
        while idx + self.seq_len <= self.emg.shape[0]:
            # check if same session
            info_start = self.info[idx]
            info_end = self.info[idx + self.seq_len - 1]

            is_same_session = info_start["subj"] == info_end["subj"]
            is_same_session &= info_start["daytesting"] == info_end["daytesting"]
            is_same_session &= info_start["time"] == info_end["time"]

            if not is_same_session:
                idx += self.seq_stride
                continue

            seq_map.append((idx, idx + self.seq_len))
            idx += self.seq_stride

        return np.array(seq_map)

    def _get_sequence(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Get a sequence.

        Args:
            idx (int): Index of the sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: EMG, stimulus and info of the sequence.
        """
        start, end = self.seq_map[idx]

        return self.emg[start:end], self.stimulus[start:end], self.info[start]

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

    def _get_pos_map(self):
        pos_map = {}

        if self.positive_mode == "none":
            return pos_map

        for idx in range(self.seq_map.shape[0]):
            start = self.seq_map[idx][0]

            info = self.info[start]

            subj = info["subj"]
            day = info["daytesting"]
            time = info["time"]
            label = info["label"]

            if self.positive_mode == "subject":
                key = f"{subj}-{label}"
            elif self.positive_mode == "session":
                key = f"{subj}-{day}-{time}-{label}"
            elif self.positive_mode == "label":
                key = f"{label}"

            if key not in pos_map:
                pos_map[key] = []

            pos_map[key].append(idx)

        return pos_map

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

        subj = info["subj"]
        day = info["daytesting"]
        time = info["time"]
        label = info["label"]

        if self.positive_mode == "subject":
            key = f"{subj}-{label}"
        elif self.positive_mode == "session":
            key = f"{subj}-{day}-{time}-{label}"
        elif self.positive_mode == "label":
            key = f"{label}"

        positive_indices = self.pos_map[key]
        positive_idx = self.rng.choice(positive_indices)

        return self._get_sequence(positive_idx)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.seq_map.shape[0]

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
        # get sequence, labels and info
        seq, labels, info_dict = self._get_sequence(idx)

        # convert sequence to blocks
        emg_blocks = self._seq_to_blocks(seq)
        stimulus_blocks = self._seq_to_blocks(labels)

        if self.positive_mode != "none":
            positive_emg, positive_stimulus, positive_info = self._sample_positive_seq(info_dict)
            positive_emg_blocks = self._seq_to_blocks(positive_emg)
            positive_stimulus_blocks = self._seq_to_blocks(positive_stimulus)

            emg = np.stack([emg_blocks, positive_emg_blocks])
            stimulus = np.stack([stimulus_blocks, positive_stimulus_blocks])
            info = np.stack([info_dict, positive_info])
        else:
            emg = np.expand_dims(emg_blocks, 0)
            stimulus = np.expand_dims(stimulus_blocks, 0)
            info = np.expand_dims(info_dict, 0)

        info = np.array([list(i.values()) for i in info])

        return (
            torch.from_numpy(emg).float(),
            torch.from_numpy(stimulus).float(),
            torch.from_numpy(info).float(),
        )
