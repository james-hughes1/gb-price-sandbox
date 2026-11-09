"""NESO data portal loaders: day-ahead demand and wind forecasts (not actuals)."""

from __future__ import annotations

import pandas as pd

from gb_price_sandbox.config import Settings


def load_demand_forecast(
    start: pd.Timestamp, end: pd.Timestamp, settings: Settings
) -> pd.DataFrame:
    """Load day-ahead national demand forecasts.

    Args:
        start: Inclusive start of the delivery period (UTC).
        end: Exclusive end of the delivery period (UTC).
        settings: Application settings.

    Returns:
        DataFrame with ``target_time``, ``issued_at`` and ``demand_mw`` columns. ``issued_at``
        is retained so feature builders can enforce availability at forecast issue time.
    """
    raise NotImplementedError


def load_wind_forecast(start: pd.Timestamp, end: pd.Timestamp, settings: Settings) -> pd.DataFrame:
    """Load day-ahead wind generation forecasts.

    Args:
        start: Inclusive start of the delivery period (UTC).
        end: Exclusive end of the delivery period (UTC).
        settings: Application settings.

    Returns:
        DataFrame with ``target_time``, ``issued_at`` and ``wind_mw`` columns.
    """
    raise NotImplementedError
