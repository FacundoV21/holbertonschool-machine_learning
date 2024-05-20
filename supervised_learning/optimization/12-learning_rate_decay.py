#!/usr/bin/env python3
"""
    task 12
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
        Creates a TensorFlow learning rate decay operation using inverse
        time decay with stepwise reduction.
    """
    lr = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True,
        name='InverseTimeDecay'
    )

    return lr
