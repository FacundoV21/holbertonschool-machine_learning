#!/usr/bin/env python3
"""
    task 8
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
        Creates a TensorFlow optimizer for the RMSProp
        optimization algorithm.
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2, epsilon=epsilon)

    return optimizer
