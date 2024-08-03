#!/usr/bin/env python3
"""
    Task 3
"""


def matrix_transpose(matrix):
    """
        func to return the transpose of a matrix
    """
    rows = len(matrix)
    cols = len(matrix[0])

    t_matrix = [[0 for r in range(rows)] for c in range(cols)]
    for i in range(rows):
        for j in range(cols):
            t_matrix[j][i] = matrix[i][j]

    return t_matrix
