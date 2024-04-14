#!/usr/bin/env python3
"""
    Task 7
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
        func to return 2 concatenated matrices
    """
    mat3 = []
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        return [row.copy() for row in mat1] + [row.copy() for row in mat2]

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        mat3 = []
        for i in range(len(mat1)):
            mat3.append(mat1[i] + mat2[i])
        return mat3

    else:
        return None
