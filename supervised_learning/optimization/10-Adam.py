#!/usr/bin/env python3
"""
    task 10
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
        Creates a TensorFlow optimizer for the Adam optimization algorithm.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

    return optimizer
