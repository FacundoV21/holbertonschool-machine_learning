#!/usr/bin/env python3
"""
    task 0
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
        Create a confusion matrix.
    """

    m, classes = labels.shape
    confusion_matrix = np.zeros((classes, classes))

    for i in range(m):
        confusion_matrix[np.argmax(labels[i])][np.argmax(logits[i])] += 1

    return confusion_matrix
