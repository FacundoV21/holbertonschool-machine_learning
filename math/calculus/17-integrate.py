#!/usr/bin/env python3
"""
    calculates the integral of a polynomia
"""


def poly_integral(poly, C=0):
    """
        Return a new list of coefficients
        representing the integral of the polynomial
    """
    integral = [C]
    i = len(poly) - 1

    while i >= 0 and poly[i] == 0:
        poly.pop(i)
        i -= 1

    for i in range(len(poly)):
        if poly[i] % (i + 1) == 0:
            integral.append(int(poly[i] / (i + 1)))
        else:
            integral.append(poly[i] / (i + 1))
    return integral
