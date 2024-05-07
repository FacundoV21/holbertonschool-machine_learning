#!/usr/bin/env python3
"""
    Task 3
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
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
            Getter for the weights vector
        """
        return self.__W

    @property
    def b(self):
        """
            Getter for the bias
        """
        return self.__b

    @property
    def A(self):
        """
            Getter for the activated output
        """
        return self.__A

    def forward_prop(self, X):
        """
            Performs forward propagation on the neuron.
        """

        if X.shape[0] != len(self.W):
            raise ValueError("Number of features in input data does not match neuron's features")

        z = np.dot(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression.
        """

        m = Y.shape[1]
        log_probs = -Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)

        return (1 / m) * np.sum(log_probs)

    def evaluate(self, X, Y):
        """
            Evaluates the neuron's predictions
        """

        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)

        return predictions, cost
