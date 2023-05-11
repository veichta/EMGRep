"""Signal processing functionality."""

import numpy as np
from scipy.signal import hilbert, savgol_filter


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


def hilbert_envelope(emg, input_freq=2000, window_size=1, smoothe=True):
    """Performs envelope transform on signal to extract amplitude.

    First takes the abs values of the hilbert frequency output (this is a complex number I think)
    Then smoothes the result by default.

    Args:
        emg (numpy.ndarray): Array of shape (nsamples, nchannels)
        input_freq (int, optional): The assumed smoothing frequency of the signal in Hz.
          Defaults to 2KHz.
        window_size (int, optional): The buffer smoothing window of the RMS op in sec.
          Default is 12.5ms
        smoothe: (bool, optional): Whether to apply savgol smooting. Defaults to True.

    Returns:
        numpy.ndarray: Filtered signal of same shape as input
    """
    x = np.abs(hilbert(emg, axis=0))
    if smoothe:
        x = savgol_preprocess(x, input_freq, window_size)
    assert x.shape == emg.shape
    """
    from matplotlib import pyplot as plt
    plt.plot(emg[100:600, 0])
    plt.plot(x[100:600, 0])
    plt.plot(rms_preprocess(emg)[100:600, 0])
    plt.plot(savgol_preprocess(emg)[100:600, 0])
    plt.plot(ma_preprocess(emg)[100:600, 0])
    plt.legend(["raw", "hilbert+savgol (55% acc)", "rms (55% acc)", \
    "savgol (43% acc)", "MA of abs (53% acc)"])
    plt.show()
    """
    return x


def ma_preprocess(emg, input_freq=2000, window_size=12.5e-3):
    """Performs Moving Average filtering on the absolute signal. Weights are uniform.

    Args:
        emg (numpy.ndarray): Array of shape (nsamples, nchannels)
        input_freq (int, optional): The assumed frequency of the signal in Hz. Defaults to 2KHz.
        window_size (int, optional): The buffer window of the RMS op in sec. Default is 12.5ms
    Returns:
        numpy.ndarray: Filtered signal of same shape as input
    """
    # @TODO validate default input freq 2KHz
    N = int(window_size * input_freq)
    x = np.apply_along_axis(lambda x: np.convolve(np.abs(x), np.ones(N) / N, mode="full"), 0, emg)
    x = x[: -N + 1]  # throw away last samples to make the window a history buffer
    assert x.shape == emg.shape  # we don't change shape
    return x


def savgol_preprocess(emg, input_freq=2000, window_size=12.5e-3):
    """Applies a savgol filter to the input signal.

    Window size is scaled to be approx. 3x the RMS window size with same params -> result similar

    Args:
        emg (numpy.ndarray): Array of shape (nsamples, nchannels)
        input_freq (int, optional): The assumed frequency of the signal in Hz. Defaults to 2KHz.
        window_size (int, optional): The buffer window of the RMS op in sec. Default is 12.5ms
    Returns:
        numpy.ndarray: Filtered signal of same shape as input
    """
    N = int(window_size * input_freq * 3)
    x = savgol_filter(emg, N, 3, axis=0)
    assert x.shape == emg.shape  # we don't change shape
    """
    from matplotlib import pyplot as plt
    print(N)
    plt.plot(emg[100:600, 0])
    plt.plot(x[100:600, 0])
    plt.plot(rms_preprocess(emg)[100:600, 0])
    plt.legend(["raw", "savgol", "rms"])
    plt.show() """

    return x
