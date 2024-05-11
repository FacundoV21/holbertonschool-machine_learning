#!/usr/bin/env python3
"""
    Deep neural network class
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
        Encodes a numeric label vector into a one-hot matrix.
    """

    m = Y.shape[0]
    if classes <= 0:
        return None

    try:
        Y = Y.reshape(-1)
    except ValueError:
        return None

    if np.any(Y < 0) or np.any(Y >= classes):
        return None

    Y_onehot = np.zeros((classes, m))
    Y_onehot[Y, np.arange(m)] = 1
    return Y_onehot
