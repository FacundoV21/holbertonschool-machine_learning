#!/usr/bin/env python3
"""
    Task 0
"""
import numpy as np


class Neuron:
    """
        This class defines a single neuron for binary classification.
    """

    def __init__(self, nx):
        """
            Initializes the neuron with the specified number of input features.
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
