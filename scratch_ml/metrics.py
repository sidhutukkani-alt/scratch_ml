"""Evaluation metrics implemented with NumPy only.

Every metric follows the usual convention ``metric(y_true, y_pred)``
and returns a plain Python ``float``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy_score",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
]


def _check_targets(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Ravel both targets to 1-D arrays and verify equal lengths."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length, got "
            f"{y_true.shape[0]} and {y_pred.shape[0]}."
        )
    return y_true, y_pred


def accuracy_score(y_true, y_pred) -> float:
    """Fraction of samples whose predicted label matches the true label.

    ``accuracy = (1 / n) * sum_i [y_true_i == y_pred_i]``
    """
    y_true, y_pred = _check_targets(y_true, y_pred)
    if y_true.size == 0:
        raise ValueError("y_true is empty.")
    return float(np.mean(y_true == y_pred))


def mean_squared_error(y_true, y_pred) -> float:
    """Mean squared error: ``MSE = (1 / n) * sum_i (y_i - y_hat_i)^2``."""
    y_true, y_pred = _check_targets(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred) -> float:
    """Mean absolute error: ``MAE = (1 / n) * sum_i |y_i - y_hat_i|``."""
    y_true, y_pred = _check_targets(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination.

    ``R^2 = 1 - SS_res / SS_tot`` where ``SS_res = sum (y - y_hat)^2``
    and ``SS_tot = sum (y - mean(y))^2``.  1.0 is a perfect fit.
    """
    y_true, y_pred = _check_targets(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0 if ss_res > 0.0 else 1.0
    return 1.0 - ss_res / ss_tot
