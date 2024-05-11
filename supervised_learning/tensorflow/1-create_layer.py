#!/usr/bin/env python3
"""
    Tensorflow
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a single layer in a TensorFlow neural network.
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=initializer,
                                  name='layer')(prev)

    return layer