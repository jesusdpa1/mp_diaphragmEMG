"""
Author: jpenalozaa
Description: functions containing different linear envelope algorithms
"""

from .filters import (
    butter_lowpass_filter,
    rolling_mean,
    rolling_rms,
)

import numpy as np
from scipy import signal
from scipy.spatial.distance import cdist
from typing import Literal


# def mean_envelope(
#     tsx: np.ndarray, window_size: int, rectify_method: Literal["power", "abs"] = "power"
# ) -> np.ndarray:
#     """
#     Calculate the linear envelop using the moving average

#     Args:
#         tsx (np.ndarray): Input signal
#         window_size (int): Window size
#         rectify_method (str, optional): Rectification method. Defaults to "power".

#     Returns:
#         np.ndarray: LE moving average
#     """
#     if rectify_method == "power":
#         tsx_rectified = np.power(tsx, 2)
#     elif rectify_method == "abs":
#         tsx_rectified = np.abs(tsx)
#     else:
#         raise ValueError("Invalid rectify method. Must be 'power' or 'abs'.")
#     return rolling_mean(tsx_rectified, window_size)


# def rms_envelope(
#     tsx: np.ndarray, window_size: int, rectify_method: Literal["power", "abs"] = "power"
# ) -> np.ndarray:
#     """
#     Calculate the linear envelop using the moving RMS

#     Args:
#         tsx (np.ndarray): Input signal
#         window_size (int): Window size
#         rectify_method (str, optional): Rectification method. Defaults to "power".

#     Returns:
#         np.ndarray: LE moving RMS
#     """
#     if rectify_method == "power":
#         tsx_rectified = np.power(tsx, 2)
#     elif rectify_method == "abs":
#         tsx_rectified = np.abs(tsx)
#     else:
#         raise ValueError("Invalid rectify method. Must be 'power' or 'abs'.")
#     return rolling_rms(tsx_rectified, window_size)


# =================== improved implementation ==================


# rectification methods
def rectify_tsx(tsx: np.ndarray, rectify_method: str) -> np.ndarray:
    """
    Rectify a time series signal using various methods.

    Parameters:
    tsx (np.ndarray): Input time series signal.
    rectify_method (str): Rectification method. Options: "abs", "power", "hilbert".

    Returns:
    np.ndarray: Rectified signal.

    Raises:
    ValueError: If rectify_method is invalid.
    """
    rectify_methods = {
        "abs": np.abs,
        "square": np.square,
        "hilbert": lambda x: np.abs(signal.hilbert(x)),
    }

    if rectify_method not in rectify_methods:
        raise ValueError(f"Invalid rectify_method: {rectify_method}")

    return rectify_methods[rectify_method](tsx)


# mean envelope methods
def get_mean_envelope(
    tsx: np.ndarray,
    overlap: float,
    window_type: str = "rectangular",
    window_size: int = 1,
    window_beta: int = None,
    rectify_method: str = "abs",
):
    """
    Moving Mean Detector implementation using convolution.

    Parameters:
    tsx (np.ndarray): Input signal.
    overlap (float): Overlap percentage (<1).
    window_size (int): Window size in samples.
    window_beta (int, optional): Beta parameter for Kaiser window. Defaults to None.
    window_type (str): Window type ('rectangular', 'hann', 'hamming', 'blackman', 'kaiser').
    rectify_method (str): Rectification method ('abs', 'square', 'hilbert')

    Returns:
    envelope_signal (array): Output of moving mean detector.
    """

    valid_rectify_methods = ["abs", "square", "hilbert"]
    if rectify_method not in valid_rectify_methods:
        raise ValueError(f"Invalid rectify method. Options: {valid_rectify_methods}")

    # rectify signal
    rectified_signal = rectify_tsx(tsx, rectify_method)

    # Convolve the signal with the squared window
    envelope_signal = rolling_mean(
        rectified_signal,
        overlap,
        window_type,
        window_size,
        window_beta,
    )
    return envelope_signal


def get_rms_envelope(
    tsx: np.ndarray,
    overlap: float,
    window_type: str = "rectangular",
    window_size: int = 1,
    window_beta: int = None,
) -> np.ndarray:
    """
    Moving RMS Detector implementation using convolution.
    rectify_method is integrated in the rms -> np.square()

    Parameters:
    tsx (np.ndarray): Input signal.
    overlap (float): Overlap percentage (<1).
    window_size (int): Window size in samples.
    window_beta (int, optional): Beta parameter for Kaiser window. Defaults to None.
    window_type (str): Window type ('rectangular', 'hann', 'hamming', 'blackman', 'kaiser').


    Returns:
    detected_signal (array): Output of moving RMS detector.
    """

    # rectify signal
    rectified_signal = tsx
    # Convolve the signal with the squared window
    envelope_signal = rolling_rms(
        rectified_signal,
        overlap,
        window_type,
        window_size,
        window_beta,
    )
    return envelope_signal


# ===================


def lp_envelope(
    tsx: np.ndarray,
    fc: float,
    fs: float,
    rectify_method: str = "abs",
) -> np.ndarray:
    """
    Calculate the linear envelop using a Low pass filter

    Args:
        tsx (np.ndarray): Input signal
        fc (float): Cut-off frequency
        rectify_method (str): Rectification method ('abs', 'square', 'hilbert')

    Returns:
        np.ndarray: LE moving RMS
    """

    valid_rectify_methods = ["abs", "square", "hilbert"]
    if rectify_method not in valid_rectify_methods:
        raise ValueError(f"Invalid rectify method. Options: {valid_rectify_methods}")

    # rectify signal
    rectified_signal = rectify_tsx(tsx, rectify_method)
    return butter_lowpass_filter(rectified_signal, fc, fs)


def tdt_envelope(tsx: np.ndarray, fc: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Extracts the envelope of a time series signal using a series of processing steps.

    function from TDT Unary Signal Processing

        y = SqrRoot2( Smooth2 (Sqrt1(Biq1-BP(x))))

            Biq1-BP(x): Band-pass filtering

            Sqrt1(): square root.

            Smooth2(): smoothing operation, Smoothing can be done using methods like moving average, Gaussian smoothing, etc.

            SqrRoot2(): This seems to be another square root operation, possibly to prepare the smoothed signal for envelope extraction.

            y: This is the final envelope of the EMG signal.

    1. notch filter
    2. bandpass filter
    3. rectify
    4. lowpass filter
    5. exponential smoothing
    6. substract mean or median

    Args:
        tsx (array-like): Input time series data.
        fc (float): Cutoff frequency (Hz).
        fs (float): Sampling frequency (Hz).
        order (int): Filter order. Minimum: 2. Default: 4.

    Returns:
        array-like: Envelope of the time series (same shape as input).
    """
    # Step 1: Notch filter
    # Step 2: Bandpass filter

    # Step 3: Rectify

    tsx_rectified = np.sqrt(np.power(tsx, 2))

    # Step 4: Lowpass filter
    tsx_lowpass_filtered = butter_lowpass_filter(tsx_rectified, fc, fs, order)
    # Step 5: Exponential smoothing
    sqrt_signal = np.sqrt(np.power(tsx_lowpass_filtered, 2))
    sqrt_signal -= np.mean(sqrt_signal)
    return sqrt_signal


def chebyshev_distances(X):
    return cdist(X, X, metric="chebyshev")


def fsampen(tsx: np.ndarray, dim: int, r: float):
    """
    Calculate the sample entropy of a time series using Chebyshev distance.

    Parameters:
    tsx (array): Time series data.
    dim (int): Embedding dimension.
    r (float): Tolerance value.

    Returns:
    float: Sample entropy of the time series.
    """

    # Create matrix of tsx by embedding dimension
    tsx_matrix = np.full((len(tsx) - dim, dim + 1), np.nan)
    tsx_matrix[:] = np.lib.stride_tricks.sliding_window_view(tsx, (dim + 1))[:, ::-1]

    # Calculate pairwise Chebyshev distances
    matrix_B = chebyshev_distances(tsx_matrix[:, :dim])
    matrix_A = chebyshev_distances(tsx_matrix[:, 1 : dim + 1])

    B = np.sum(matrix_B.flatten() <= r)
    A = np.sum(matrix_A.flatten() <= r)

    # Calculate ratio of natural logarithm, with normalization correction
    if B == 0:
        result = np.inf
    else:
        result = -np.log(A / B)

    # Correct inf to a maximum value
    if np.isinf(result):
        result = -np.log(2 / ((len(tsx) - dim - 1) * (len(tsx) - dim)))
    return result
