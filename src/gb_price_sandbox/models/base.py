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
