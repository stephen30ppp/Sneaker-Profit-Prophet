"""
Model Evaluation Metrics

Computes MSE, RMSE, MAE, and directional accuracy
for model vs baseline comparison.
"""
import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of times the model correctly predicts
    the direction of price movement.
    """
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return float(np.mean(true_direction == pred_direction))
