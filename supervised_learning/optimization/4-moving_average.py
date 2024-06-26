#!/usr/bin/env python3
"""
    task 4
"""


def moving_average(data, beta):
    """
        Calculates the weighted moving average of a dataset with bias correction.
    """
    moving_averages = []
    a=0

    for i in range(len(data)):
        a = beta * a + (1 - beta) * data[i]
        b = 1 - beta ** (i + 1)
        moving_average = a / b
        moving_averages.append(moving_average)

    return moving_averages
