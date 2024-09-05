"""
Author: jpenalozaa
Description: functions containing different stats function helpers
"""

import numpy as np
from typing import Literal


def dc_removal(tsx: np.ndarray, method: Literal["mean", "median"]):
    """
    Remove the DC component from a time series signal.

    Args:
        tsx (np.ndarray): Input time series signal.
        method (Literal["mean", "median"]): Method to use for DC removal.

    Returns:
        np.ndarray: Time series signal with DC component removed.

    Raises:
        ValueError: If an invalid method is provided.
    """
    if method == "mean":
        return tsx - np.mean(tsx)
    if method == "median":
        return tsx - np.median(tsx)
    else:
        raise ValueError("Invalid method. Choose 'mean' or 'median'.")


def min_max_normalization(tsx: np.ndarray) -> np.ndarray:
    """
    Normalize tsx using min-max normalization.

    Args:
    - tsx (numpy array): Input tsx to be normalized.

    Returns:
    - normalized_tsx (numpy array): Normalized tsx.
    """
    min_val = np.min(tsx)
    max_val = np.max(tsx)
    normalized_tsx = (tsx - min_val) / (max_val - min_val)
    return normalized_tsx
