#!/usr/bin/env python3
import numpy as np
"""
    Task 12
"""


def np_elementwise(mat1, mat2):
    """
        func performs element-wise addition, subtraction, multiplication, and division
    """
    elementwise_sum = mat1 + mat2
    
    elementwise_diff = mat1 - mat2
    
    elementwise_product = mat1 * mat2
    
    elementwise_quotient = np.divide(mat1, mat2, out=np.zeros_like(mat1), where=mat2!=0)
    
    return elementwise_sum, elementwise_diff, elementwise_product, elementwise_quotient
