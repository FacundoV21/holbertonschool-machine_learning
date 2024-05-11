#!/usr/bin/env python3
"""
    Tensorflow
"""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
        Creates TensorFlow placeholders for input data (x) and labels (y).
    """

    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
