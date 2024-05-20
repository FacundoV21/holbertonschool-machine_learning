#!/usr/bin/env python3
"""
    task 14
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.
    """
    x = tf.keras.layers.Dense(
        n, kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'))(prev)

    gamma = tf.Variable(tf.ones(shape=(n,)), name='gamma', trainable=True)
    beta = tf.Variable(tf.zeros(shape=(n,)), name='beta', trainable=True)
    epsilon = 1e-7

    batch_mean, batch_var = tf.nn.moments(x, axes=[0], keepdims=True)
    normalized = (x - batch_mean) / tf.sqrt(batch_var + epsilon)

    out = gamma * normalized + beta

    if activation is not None:
        out = activation(out)

    return out
