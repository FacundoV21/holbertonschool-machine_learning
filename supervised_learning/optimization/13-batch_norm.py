#!/usr/bin/env python3
"""
    task 13
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
        Performs batch normalization on a numpy array.
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True) + epsilon
    Z_norm = (Z - mean) / np.sqrt(var)
    Z_norm = Z_norm * gamma + beta

    return Z_norm
