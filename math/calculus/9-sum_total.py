#!/usr/bin/env python3
"""
    task 9
"""


def summation_i_squared(n):
    """
        This function calculates the sum of squares from 0 to n (inclusive) using recursion.
    """

    if n < 0:
        return None

    if n == 0:
        return 0

    return n**2 + summation_i_squared(n-1)
