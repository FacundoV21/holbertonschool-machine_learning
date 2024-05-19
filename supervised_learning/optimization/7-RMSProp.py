#!/usr/bin/env python3
"""
    task 7
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
        Updates a variable and its second moment using RMSProp
        optimization.
    """

    s_t = beta2 * s + (1 - beta2) * np.square(grad)
    s_t_sqrt = np.sqrt(s_t) + epsilon
    var_updated = var - alpha * grad / s_t_sqrt

    return var_updated, s_t
