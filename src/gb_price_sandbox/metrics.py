"""Forecast evaluation metrics: MAE, pinball loss, interval coverage, CRPS (from quantiles)."""

from __future__ import annotations

from collections.abc import Sequence

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
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def interval_coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Fraction of observations falling inside ``[lower, upper]`` (inclusive).

    For a 10%-90% interval a calibrated forecast should score about 0.8.
    """
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def crps_from_quantiles(
    y_true: ArrayLike,
    q_pred: ArrayLike,
    quantiles: Sequence[float],
) -> float:
    """Approximate mean CRPS from a set of predicted quantiles.

    Uses the identity CRPS = 2 * integral over tau in (0, 1) of the pinball loss at tau,
    approximated by the average pinball loss over the supplied quantile levels:
    ``CRPS ~= (2 / K) * sum_k pinball(tau_k)``. With few quantiles (e.g. 0.1/0.5/0.9) this is a
    coarse approximation, but it is consistent across models, which is what a leaderboard needs.

    Args:
        y_true: Observed values, shape ``(n,)``.
        q_pred: Predicted quantiles, shape ``(n, K)``, columns ordered as ``quantiles``.
        quantiles: Quantile levels, length ``K``, each in (0, 1).

    Returns:
        Approximate mean CRPS.
    """
    q_arr = np.asarray(q_pred, dtype=float)
    if q_arr.ndim != 2 or q_arr.shape[1] != len(quantiles):
        raise ValueError(f"q_pred must have shape (n, {len(quantiles)}), got {q_arr.shape}")
    losses = [pinball_loss(y_true, q_arr[:, k], q) for k, q in enumerate(quantiles)]
    return float(2.0 * np.mean(losses))
