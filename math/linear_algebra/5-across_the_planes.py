#!/usr/bin/env python3
"""
    Task 5
"""


def add_matrices2D(mat1, mat2):
    """
        func to return the sum of 2 matrices
    """

    rows = len(mat1)
    cols = len(mat1[0])

    if rows != len(mat2) or cols != len(mat2[0]):
        return None

    sum_matrix = [[0 for r in range(rows)] for c in range(cols)]
    for i in range(rows):
        for j in range(cols):
            sum_matrix[i][j] = mat1[i][j] + mat2[i][j]

    return sum_matrix
