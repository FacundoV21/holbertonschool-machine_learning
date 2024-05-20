#!/usr/bin/env python3
"""
    task 9
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
        Updates a variable, its first moment, and second moment
        using the Adam optimization algorithm.
    """
    v_t = beta1 * v + (1 - beta1) * grad
    s_t = beta2 * s + (1 - beta2) * grad * grad
    v_corrected = v_t / (1 - beta1**t)
    s_corrected = s_t / (1 - beta2**t)
    var_updated = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var_updated, v_t, s_t
