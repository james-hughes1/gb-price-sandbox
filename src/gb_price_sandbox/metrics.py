"""Forecast evaluation metrics: MAE, pinball loss, interval coverage, CRPS (from quantiles)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, q: float) -> float:
    """Mean pinball (quantile) loss at quantile level ``q``.

    Args:
        y_true: Observed values.
        y_pred: Predicted ``q``-quantile.
        q: Quantile level in (0, 1).

    Returns:
        Mean pinball loss.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))
