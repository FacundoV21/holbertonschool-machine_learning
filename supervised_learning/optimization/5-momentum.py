#!/usr/bin/env python3
"""
    task 5
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
        Updates a variable and its momentum using
        gradient descent with momentum.
    """
    v_t = beta1 * v + (1 - beta1) * grad

    var_updated = var - alpha * v_t

    return var_updated, v_t
