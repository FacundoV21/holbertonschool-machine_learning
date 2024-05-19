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
    num_batches = int(np.ceil(m / batch_size))

    X, Y = shuffle_data.shuffle_data(X, Y)

    mini_batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, m)

    X_batch = X[start_idx:end_idx]
    Y_batch = Y[start_idx:end_idx]
    mini_batches.append((X_batch, Y_batch))

    return mini_batches
