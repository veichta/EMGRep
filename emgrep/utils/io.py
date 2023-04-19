"""Helper functions for loading and saving data."""

import os
from typing import Any

import scipy.io as sio


def load_mat(filename: str) -> dict[str, Any]:
    """Load a .mat file.

    Args:
        filename (str): Path to the .mat file.

    Returns:
        dict[str, Any]: Dictionary containing the variables in the .mat file.
    """
    return sio.loadmat(filename)


def get_recording(subject: int, day: int, time: int, data_path: str) -> dict[str, Any]:
    """Load a recording from the dataset.

    Args:
        subject (int): Id of the subject (1-10).
        day (int): Day of the recording (1-5).
        time (int): Time of the recording (1-2).
        data_path (str): Path to the dataset.

    Raises:
        ValueError: If subject, day or time are not in the valid range.

    Returns:
        dict[str, Any]: Dictionary containing the variables in the .mat file.
    """
    if subject not in range(1, 11):
        raise ValueError(f"Subject {subject} not in [1, 10]")

    if time not in range(1, 3):
        raise ValueError(f"Time {time} not in [1, 2]")

    if day in range(1, 6):
        fpath = os.path.join(data_path, f"S{subject}_D{day}_T{time}.mat")
    else:
        raise ValueError(f"Day {day} not in [1, 2, 3, 4, 5]")
    return load_mat(fpath)
