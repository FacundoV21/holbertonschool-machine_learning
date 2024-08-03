#!/usr/bin/env python3
"""
    Task 2
"""


def matrix_shape(matrix):
    """
        func to return the size of a matrix
    """
    if isinstance(matrix, list):
        return [len(matrix)] + matrix_shape(matrix[0])
    else:
        return []
