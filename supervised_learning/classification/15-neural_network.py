#!/usr/bin/env python3
"""
    Neural network class
"""

import numpy as np


class NeuralNetwork:
    """
        Neural network class with one hidden layer for binary classification.
    """

    def __init__(self, nx, nodes):
        """
            Init function.
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter for the weights vector of the hidden layer.
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for the bias of the hidden layer.
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for the activated output of the hidden layer.
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for the weights vector of the output neuron.
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for the bias of the output neuron.
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for the activated output of the output neuron (prediction).
        """
        return self.__A2

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network.
        """

        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        """

        m = Y.shape[1]
        log_probs = -Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)
        return (1 / m) * np.sum(log_probs)

    def evaluate(self, X, Y):
        """
            Evaluates the neural network's predictions.
        """

        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        results = np.where(A2 >= 0.5, 1, 0)
        return results, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neural network
        """

        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.dot(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
            Trains the neural network.
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []

        A1, A2 = self.forward_prop(X)
        costs.append(self.cost(Y, A2))

        for i in range(iterations):
            self.gradient_descent(X, Y, A1, A2, alpha)
            A1, A2 = self.forward_prop(X)

            if verbose and (i % step == 0 or i == iterations - 1):
                cost = self.cost(Y, A2)
                print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)

        if graph:
            import matplotlib.pyplot as plt

            plt.plot(range(len(costs)), costs, label="Training Cost")
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.legend()
            plt.show()

        return self.evaluate(X, Y)
