#!/usr/bin/env python3
"""
    Task 8
"""


def mat_mul(mat1, mat2):
    """
        func to return 2 multiplicated matrices
    """
    cols = len(mat1[0])
    rows = len(mat2)

    if cols != rows:
        return None

    mat3 = [[0 for c in range(len(mat2[0]))] for r in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                mat3[i][j] += mat1[i][k] * mat2[k][j]

    return mat3
