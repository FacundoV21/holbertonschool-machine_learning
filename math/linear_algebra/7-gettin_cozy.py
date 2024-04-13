#!/usr/bin/env python3
"""
    Task 6
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

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
print(mat4)
print(mat5)
mat1[0] = [9, 10]
mat1[1].append(5)
print(mat1)
print(mat4)
print(mat5)