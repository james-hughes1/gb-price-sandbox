"""Common interface for day-ahead price forecasters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def quantile_column(q: float) -> str:
    """Column name for quantile ``q`` (e.g. 0.1 -> ``"q10"``); strings keep parquet happy."""
    return f"q{round(q * 100):02d}"


QUANTILE_COLUMNS: tuple[str, ...] = tuple(quantile_column(q) for q in QUANTILES)


@runtime_checkable
class Forecaster(Protocol):
    """A model that can be refit in a walk-forward loop and emit quantile forecasts."""

    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster:  # noqa: N803
        """Fit on features ``X`` and target prices ``y``; return self."""
        ...

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Predict quantiles for each row of ``X``.

        Returns:
            DataFrame indexed like ``X`` with one column per entry of ``QUANTILE_COLUMNS``.
            Point forecasters return the same value in every column.
        """
        ...


def point_as_quantiles(point: pd.Series) -> pd.DataFrame:
    """Wrap a point forecast as a degenerate quantile forecast (all quantiles equal)."""
    values = np.repeat(point.to_numpy(dtype=float)[:, None], len(QUANTILE_COLUMNS), axis=1)
    return pd.DataFrame(values, index=point.index, columns=list(QUANTILE_COLUMNS))


def sort_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort each row's quantiles into increasing order.

    Quantile models fitted separately per quantile can *cross* (q10 above q50). Sorting
    ("rearrangement", Chernozhukov et al., 2010) fixes this and never makes pinball loss worse.
    Rows with a missing value are left untouched.
    """
    cols = list(QUANTILE_COLUMNS)
    values = frame[cols].to_numpy(dtype=float, copy=True)
    ok = ~np.isnan(values).any(axis=1)
    values[ok] = np.sort(values[ok], axis=1)
    out = frame.copy()
    out[cols] = values
    return out


@runtime_checkable
class HistoryForecaster(Protocol):
    """A time-series model that also needs the latest published prices when it forecasts.

    The walk-forward loop still calls :meth:`fit` weekly on the training window. Then, for each
    delivery day D, it calls :meth:`predict_day` with every price published by the D-1 issue
    time, so the model can update on the morning of D-1 without being refitted.
    """

    name: str
    uses_history: bool

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HistoryForecaster:  # noqa: N803
        """Fit on features ``X`` and target prices ``y`` (both indexed by period start)."""
        ...

    def predict_day(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast every period in ``future``.

        Args:
            history: Features plus ``price`` for every period whose price was published by
                the issue time (a regular half-hourly grid; missing prices are NaN).
            future: Features for the periods from the first unpublished one to the end of
                delivery day D.

        Returns:
            Quantile frame indexed like ``future``.
        """
        ...
