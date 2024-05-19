#!/usr/bin/env python3
"""
    task 3
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
        Creates mini-batches for training a neural network using
        mini-batch gradient descent.
    """

    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be NumPy\
                                arrays with the same number of data points.")

    m = X.shape[0]
    num_batches = len(X) // batch_size
    btchSizcpy = batch_size
    X, Y = shuffle_data(X, Y)
    i = 0
    mini_batches = []

    while i in range(len(X)):
        tup = (X[i:btchSizcpy], Y[i: btchSizcpy])
        mini_batches.append(tup)
        i += batch_size
        btchSizcpy += batch_size

    return mini_batches
