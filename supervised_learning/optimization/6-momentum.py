#!/usr/bin/env python3
"""
    task 6
"""
import numpy as np
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
        Creates a TensorFlow optimizer for gradient descent with momentum.
    """

    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    return optimizer
