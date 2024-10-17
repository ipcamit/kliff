import numpy as np


def MSE_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray = 1.0,
) -> np.ndarray:
    r"""
    Compute the mean squared error (MSE) of the residuals.

    Args:

    Returns:
        The MSE of the residuals.
    """
    residuals = (predictions - targets) * weights
    return np.mean(residuals**2)
