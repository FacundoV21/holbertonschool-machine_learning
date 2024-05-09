#!/usr/bin/env python3
"""
    Task 3
"""
import numpy as np
import matplotlib.pyplot as plt


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

        self.__W = np.random.normal(size=(1, nx))
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

        mtx = np.matmul(self.__W, X) + self.__b
        res = (1 / (1 + np.exp(-mtx)))
        self.__A = res
        return res

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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron.
        """

        m = X.shape[1]
        dZ = A - Y

        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
             graph=True, step=100):
        """
            Trains the neuron using gradient descent.
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        self.__A = self.forward_prop(X)
        step2 = step
        costs = []
        cost = self.cost(Y, self.__A)
        print(f"Cost after 0 iterations: {cost}")
        for i in range(iterations + 1):
            self.gradient_descent(X, Y, self.__A, alpha)
            if verbose is True and i == step2:
                print(f"Cost after {i} iterations: {self.cost(Y, self.__A)}")
                step2 += step
            if graph is True:
                costs.append(self.cost(Y, self.__A))
            self.__A = self.forward_prop(X)
        if graph is True:
            plt.scatter(np.arange(0, iterations + 1), np.array(costs))
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()

        return self.evaluate(X, Y)
