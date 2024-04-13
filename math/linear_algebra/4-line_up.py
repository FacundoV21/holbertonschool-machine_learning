#!/usr/bin/env python3
"""
    Task 4
"""


def add_arrays(arr1, arr2):
    """
        func to return the sum of 2 arrays
    """

    if len(arr1) != len(arr2):
        return None

    arr3 = [0 for r in range(len(arr1))]
    for i in range(len(arr1)):
        arr3[i] = arr1[i] + arr2[i]
    return arr3
