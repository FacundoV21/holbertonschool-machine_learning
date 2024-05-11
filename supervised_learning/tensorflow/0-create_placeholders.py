#!/usr/bin/env python3
"""
    Tensorflow
"""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
        Creates TensorFlow placeholders for input data (x) and labels (y).
    """

    x = tf.placeholder(tf.float32, shape=(nx, None), name="x")  # Placeholder for input data
    y = tf.placeholder(tf.float32, shape=(classes, None), name="y")  # Placeholder for labels

    return x, y
