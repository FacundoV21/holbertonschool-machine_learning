#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
    task 0
"""


def line():
    """
        func to plot y as a line graph
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color='red')
    plt.xlim(0, 10)
    plt.show()
