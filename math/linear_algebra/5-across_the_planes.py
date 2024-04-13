#!/usr/bin/env python3
"""
    Task 5
"""


def add_matrices2D(mat1, mat2):
    """
        func to return the sum of 2 matrices
    """

    rows1 = len(mat1)
    cols1 = len(mat1[0])
    rows2 = len(mat2)

    if rows1 != rows2:
        return None

    for r in range(rows1):
        if len(mat1[i]) != len(mat2[i]):
            return None

    sum_matrix = [[0 for r in range(rows1)] for c in range(cols1)]
    for i in range(rows1):
        for j in range(cols1):
            sum_matrix[i][j] = mat1[i][j] + mat2[i][j]

    return sum_matrix
