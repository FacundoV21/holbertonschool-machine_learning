#!/usr/bin/env python3
"""
    task 0
"""
import numpy as np


def normalization_constants(X):
    """
        that calculates the normalization (standardization)
        constants of a matrix
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std[std == 0] = 1e-8

    return mean, std
