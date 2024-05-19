#!/usr/bin/env python3
"""
    task 2
"""
import numpy as np


def shuffle_data(X, Y):
    """
        Shuffles the data points in two NumPy arrays in the same way.
    """

    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be\
                                NumPy arrays with the same\
                                number of data points.")

    m = X.shape[0]

    permutation = np.random.permutation(m)

    return X[permutation], Y[permutation]
