#!/usr/bin/env python3
"""
    Deep neural network class
"""

import numpy as np


class DeepNeuralNetwork:
    """
        Deep neural network class performing binary classification.
    """

    def __init__(self, nx, layers):
        """
            Initializes the deep neural network.
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        for num_nodes in layers:
            if not isinstance(num_nodes, int) or num_nodes <= 0:
                raise ValueError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i, num_nodes in enumerate(layers, start=1):
            if i == 1:
                prev_nodes = nx
            else:
                prev_nodes = layers[i - 2]
            fan_in = prev_nodes
            self.weights[f"W{i}"] = np.random.randn(num_nodes, prev_nodes) * np.sqrt(2 / fan_in)
            self.weights[f"b{i}"] = np.zeros((num_nodes, 1))

        self.cache["A0"] = np.zeros((nx, 1))
