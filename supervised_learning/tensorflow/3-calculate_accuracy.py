#!/usr/bin/env python3
"""
    Tensorflow
"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of the network's predictions.
    """

    y = tf.cast(y, tf.bool)
    y_pred = tf.cast(y_pred > 0.5, tf.bool)

    correct_predictions = tf.cast(tf.equal(y, y_pred), tf.float32)
    accuracy = tf.reduce_mean(correct_predictions)

    return accuracy
