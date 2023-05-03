"""Signal processing functionality."""

import numpy as np


def rms_preprocess(emg, input_freq=2000, window_size=12.5e-3):
    """Performs RMS preprocessing according to eq 4.1 in the Thesis.

    Args:
        emg (numpy.ndarray): Array of shape (nsamples, nchannels)
        input_freq (int, optional): The assumed frequency of the signal in Hz. Defaults to 2KHz.
        window_size (int, optional): The buffer window of the RMS op in sec. Default is 12.5ms
    Returns:
        numpy.ndarray: Filtered signal of same shape as input
    """
    # @TODO validate default input freq 2KHz
    N = int(window_size * input_freq)
    x = np.apply_along_axis(lambda x: np.convolve(x**2, np.ones(N) / N, mode="full"), 0, emg)
    x = np.sqrt(x)  # root of formula 4.1
    x = x[: -N + 1]  # throw away last samples to make the window a history buffer
    assert x.shape == emg.shape  # we don't change shape
    return x
