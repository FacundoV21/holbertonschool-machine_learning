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

        if all(map(lambda n: isinstance(n, int) and n > 0, layers)):
            self.__L = len(layers)
            self.__cache = {}
            self.__weights = {}
            for i in range(self.__L):
                self.__weights[f'W{i + 1}'] = np.random.normal(size=(
                    layers[i], nx))*(np.sqrt(2 / nx))
                self.__weights[f'b{i + 1}'] = np.zeros((layers[i], 1))
                nx = layers[i]

        else:
            raise TypeError('layers must be a list of positive integers')

    @property
    def L(self):
        """
            Getter for the number of layers in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """
            Getter for the cache dictionary.
        """
        return self.__cache

    @property
    def weights(self):
        """
            Getter for the weights dictionary.
        """
        return self.__weights

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network.
        """

        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            prev_activation = self.__cache[f"A{i - 1}"]
            W = self.__weights[f"W{i}"]
            b = self.__weights[f"b{i}"]
            z = np.dot(W, prev_activation) + b
            self.__cache[f"A{i}"] = 1 / (1 + np.exp(-z))

        return self.__cache[f"A{self.__L}"], self.__cache
