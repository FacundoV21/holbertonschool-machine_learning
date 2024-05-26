#!/usr/bin/env python3
"""
    task 2
"""
import numpy as np


def precision(confusion):
    """
        Calculates the precision for each class in a confusion matrix.
    """
    precision = []
    a = confusion.copy()
    for i in range(confusion.shape[0]):
        precision.append(confusion[i][i] / confusion.T[i].sum())

    return np.array(precision)
