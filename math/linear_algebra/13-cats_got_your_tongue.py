#!/usr/bin/env python3
import numpy as np
"""
    Task 13
"""


def np_cat(mat1, mat2, axis=0):
    """
        func that concatenates two matrices
    """
    return np.concatenate((mat1, mat2), axis=axis)
