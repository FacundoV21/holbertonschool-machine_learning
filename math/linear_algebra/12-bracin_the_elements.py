#!/usr/bin/env python3
"""
    Task 12
"""


def np_elementwise(mat1, mat2):
    """
        func performs element-wise addition, subtraction, multiplication, and division
    """
    elementwise_sum = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]

    elementwise_dif = [[mat1[i][j] - mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]

    elementwise_mult = [[mat1[i][j] * mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]

    elementwise_div = [[mat1[i][j] / mat2[i][j] if mat2[i][j] != 0 else 0 for j in range(len(mat1[0]))] for i in range(len(mat1))]

    return elementwise_sum, elementwise_dif, elementwise_mult, elementwise_div
