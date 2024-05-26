#!/usr/bin/env python3
"""
    task 1
"""
import numpy as np


def sensitivity(confusion):
    """
        Calculates the sensitivity (recall) for each class in a
        confusion matrix.
    """

    classes = confusion.shape[0]

    sensitivity = np.zeros(classes)
    for i in range(classes):
        true_positives = confusion[i, i]
        false_negatives = np.sum(confusion[i, :]) - true_positives

        if true_positives == 0:
            sensitivity[i] = 0
        else:
            sensitivity[i] = true_positives / (
                true_positives + false_negatives)

    return sensitivity
