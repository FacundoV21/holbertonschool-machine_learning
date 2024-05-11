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

    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None

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

def one_hot_decode(one_hot):
    """
        Decodes a one-hot encoded matrix into a vector of labels.
    """

    if one_hot is None:
        return None
    if type(one_hot) is not np.ndarray:
        return None

    try:
        m = one_hot.shape[1]
        classes = one_hot.shape[0]

        if len(one_hot.shape) != 2 or one_hot.sum(axis=0).mean() != 1:
            return None

        labels = np.argmax(one_hot, axis=1)

        return labels

    except Exception as ex:
        return None
