#!/usr/bin/env python3
"""
    Tensorflow
"""

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
        Creates the forward propagation graph for a neural network.
    """

    if len(layer_sizes) != len(activations):
        raise ValueError(
            "layer_sizes and activations must have the same length")

    current_layer = x

    for i, (n, activation) in enumerate(zip(layer_sizes, activations)):
        current_layer = create_layer(current_layer, n, activation)

    return current_layer
