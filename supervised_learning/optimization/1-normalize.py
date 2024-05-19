#!/usr/bin/env python3
"""
    task 1
"""
import numpy as np


def normalize(X, m, s):
    """
        Normalizes (standardizes) a NumPy array using
        the given mean and standard deviation.
    """

    if not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray):
        if not isinstance(s, np.ndarray):
            raise TypeError("Input must be NumPy arrays.")

    if X.shape[1] != m.shape[0] or X.shape[1] != s.shape[0]:
        raise ValueError("Shapes of X, m, and s must be compatible.")

    s[s == 0] = 1e-8

    return (X - m) / s
