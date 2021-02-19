import numpy as np

from numpy import ndarray as Array


def sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: Array) -> Array:
    return x * (1 - x)