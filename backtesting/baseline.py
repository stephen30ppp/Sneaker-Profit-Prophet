"""
Naive Random Walk Baseline

Implements the random walk (naive) forecast as a benchmark:
the best prediction for tomorrow's price is today's price.
"""
import numpy as np
import pandas as pd


def random_walk_forecast(prices: np.ndarray, horizon: int) -> np.ndarray:
    """
    Generate naive random walk predictions.
    Prediction = last known price repeated for the entire horizon.

    Args:
        prices: Historical price array
        horizon: Number of days to forecast

    Returns:
        Array of shape (num_windows, horizon) with naive predictions
    """
    # TODO: Implement sliding window naive forecast
    raise NotImplementedError
