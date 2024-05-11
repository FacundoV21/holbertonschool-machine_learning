#!/usr/bin/env python3
"""
    Deep neural network class
"""

import numpy as np


def one_hot_decode(one_hot):
    """
        Decodes a one-hot encoded matrix into a vector of labels.
    """

    classes, m = one_hot.shape  # Get classes and number of examples

    if len(one_hot.shape) != 2 or one_hot.sum(axis=0).mean() != 1:
        return None

    labels = np.argmax(one_hot, axis=1)

    return labels
