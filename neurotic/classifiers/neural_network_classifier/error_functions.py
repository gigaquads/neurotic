import numpy as np

from numpy import ndarray as Array


def MSE(error: Array) -> float:
    return np.sum(error**2) / (2 * len(error))