"""Common interface for day-ahead price forecasters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


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
            DataFrame indexed like ``X`` with one column per quantile in ``QUANTILES``.
            Point forecasters should return the same value in every column.
        """
        ...
