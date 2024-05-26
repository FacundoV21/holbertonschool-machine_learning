#!/usr/bin/env python3
"""
    task 4
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
        Calculates the F1 score for each class in a confusion matrix.
    """

    classes = confusion.shape[0]

    f1 = np.zeros(classes)
    for i in range(classes):
        sens = sensitivity(confusion)[i]
        prec = precision(confusion)[i]

        epsilon = 1e-7
        denominator = sens + prec
        if denominator < epsilon:
            f1[i] = 0
        else:
            f1[i] = 2 * sens * prec / denominator

    return f1
