#!/usr/bin/env python3
"""
    task 10
"""


def poly_derivative(poly):
    """
        This function calculates the derivative of a polynomial 
        represented by a list of coefficients.
    """
    if not isinstance(poly, list):
        return None

    if len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]

    derivative = [poly[i] * (i) for i in range(1, len(poly))]
    return derivative
