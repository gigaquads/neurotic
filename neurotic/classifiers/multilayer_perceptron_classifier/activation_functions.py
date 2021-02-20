import numpy as np

from numpy import ndarray as Array


def sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: Array) -> Array:
    return x * (1 - x)


def softmax(x: Array) -> Array:
    """
    Compute softmax values for each sets of scores in x.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)