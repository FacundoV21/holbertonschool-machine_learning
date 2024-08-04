#!/usr/bin/env python3
"""
    Task 12
"""

import numpy as np

def np_elementwise(mat1, mat2):
    """
        func performs element-wise addition, subtraction, multiplication, and division
    """
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)

    elementwise_sum = np.add(mat1, mat2)
    elementwise_dif = np.subtract(mat1, mat2)
    elementwise_mult = np.multiply(mat1, mat2)
    elementwise_div = np.divide(mat1, mat2, out=np.zeros_like(mat1, dtype=float), where=mat2!=0)

    return elementwise_sum, elementwise_dif, elementwise_mult, elementwise_div