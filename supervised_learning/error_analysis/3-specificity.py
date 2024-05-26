#!/usr/bin/env python3
"""
    task 3
"""
import numpy as np


def specificity(confusion):
    """
        Calculates the specificity (true negative rate) for each
        class in a confusion matrix.
    """

    precision = []
    a = confusion.copy()
    b = confusion.copy()

    for i in range(confusion.shape[0]):
        a[i] = np.zeros(a[i].shape)
        a = a.T

        a[i] = np.zeros(a[i].shape)
        a = a.T

        b[i] = np.zeros(b[i].shape)
        precision.append(a.sum() / b.sum())
        a = confusion.copy()
        b = confusion.copy()

    return np.array(precision)
