"""
Author: jpenalozaa
Description: functions containing different filters for time series data
"""

import numpy as np
from scipy.signal import filtfilt, sosfilt, sosfiltfilt, iirnotch, butter


def moving_window(tsx, window_size, step=1):
    """
    Returns a generator that yields a moving window of size `window_size` over `tsx` with a step size of `step`.

    Args:
        tsx (list): The input tsx.
        window_size (int): The size of the moving window.
        step (int, optional): The step size. Defaults to 1.

    Yields:
        list: A moving window of size `window_size` over `tsx`.
    """
    for i in range(0, len(tsx) - window_size + 1, step):
        yield tsx[i : i + window_size]


def rolling_mean(tsx: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the moving window mean

    Args:
        tsx (np.ndarray): Input signal
        N (int): Window size

    Returns:
        np.ndarray: Moving window mean values
    """
    kernel = np.ones(window_size) / window_size
    ma = np.convolve(tsx, kernel, mode="valid")
    ma = np.pad(ma, (window_size // 2, window_size // 2), mode="constant")
    return ma


def rolling_rms(tsx: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the moving window RMS

    Args:
        tsx (np.ndarray): Input signal
        N (int): Window size

    Returns:
        np.ndarray: Moving window RMS values
    """
    kernel = np.ones(window_size) / window_size
    squared = np.square(tsx)
    rms = np.sqrt(np.convolve(squared, kernel, mode="valid"))
    rms = np.pad(rms, (window_size // 2, window_size // 2), mode="constant")
    return rms


# =================== Butter Filters ===========================


def _butter_lowpass(f0: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Private function to create a Butterworth lowpass filter.

    Args:
        f0 (float): Lower cutoff frequency.
        fs (float): Sampling frequency.
        order (int, optional): Filter order. Defaults to 4.

    Returns:
        sos (array-like): Filter coefficients.
    """
    nyq = 0.5 * fs
    normal_cutoff = f0 / nyq
    return butter(order, normal_cutoff, analog=False, btype="low", output="sos")


def butter_lowpass_filter(
    tsx: np.ndarray, f0: float, fs: float, order: int = 4
) -> np.ndarray:
    """
    Apply a lowpass filter to the time series.

    Args:
        tsx (array-like): Input time series.
        f0 (float): Cutoff frequency (Hz).
        fs (float): Sampling frequency (Hz).
        order (int, optional): Filter order. Defaults to 4.

    Returns:
        array-like: Filtered time series.
    """
    sos = _butter_lowpass(f0, fs, order)
    return sosfiltfilt(sos, tsx)


def _butter_bandpass(
    lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    """
    Private function to create a Butterworth bandpass filter.

    Args:
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        fs (float): Sampling frequency.
        order (int, optional): Filter order. Defaults to 5.

    Returns:
        sos (array-like): Filter coefficients.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def butter_bandpass_filter(
    tsx: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to a signal.

    Args:
        tsx (array-like): Input signal.
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        order (int, optional): Filter order. Defaults to 5.
        fs (float, optional): Sampling frequency. Defaults to 1.0.

    Returns:
        y (array-like): Filtered signal.
    """
    sos = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, tsx)
    return y


def notch_filter(
    tsx: np.ndarray, f0: float = 60, Q: float = 30, fs: float = 1.0
) -> np.ndarray:
    """
    Apply a notch filter to the time series.

    Args:
        tsx (array-like): Input time series tsx.
        f0 (float): Frequency to remove (Hz), center frequency.
        order (int, optional): Filter order. Defaults to 4.
        fs (int, optional): Sampling frequency (Hz). Defaults to 1.0.

    Returns:
        array-like: Filtered time series tsx.
    """
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, tsx)


def ffc_filter(
    tsx: np.ndarray, alpha: float = None, fc: float = 60.0, fs: float = 1.0
) -> np.ndarray:
    """
    Implements the FFC filter as described in [1].

    Apply the FFC filter to the input signal.

    y(k) = x(k) + (alpha * x)(k-N)
    x = input signal
    N = delay expressed in number of samples fs/f(w) where w is the target frequency to remove
    alpha = regulates aspects of the filter behavior (-1)

    [1] D. Esposito, J. Centracchio, P. Bifulco, and E. Andreozzi,

    “A smart approach to EMG envelope extraction and powerful denoising for human–machine interfaces,”

    Sci. Rep., vol. 13, no. 1, p. 7768, 2023, doi: 10.1038/s41598-023-33319-4.

    Parameters:
    tsx (np.ndarray): Input signal.
    alpha (float, optional): Filter behavior regulator. Defaults to -1.0.
    fc (float, optional): Target frequency to remove. Defaults to 60.0.
    fs (float, optional): Sampling frequency. Defaults to 24000.0.

    Returns:
    np.ndarray: Filtered signal.

    """
    assert tsx.ndim == 1, "Array must be one-dimensional"

    fs_ffc = int(fs / fc)  # delay expressed in number of samples
    return tsx + alpha * np.roll(tsx, -fs_ffc)  # apply the FFC filter
