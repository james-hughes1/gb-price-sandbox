"""Point-in-time selection of forecast vintages: the main guard against look-ahead leakage.

Exogenous forecasts (demand, wind, weather) are revised many times. A day-ahead price
forecast for delivery day D, issued at D-1 ``hh:mm``, may only use the latest vintage of each
input published at or before that moment. Using a later revision, or the "historic" files
that keep only the final one, would leak information from the future into the backtest.
"""

from __future__ import annotations

import pandas as pd

from gb_price_sandbox.timing import delivery_day, ensure_utc_index, issue_time

TARGET = "target_time"
ISSUED = "issued_at"


def cutoff_times(target_times: pd.Index, hour: int = 10, minute: int = 0) -> pd.DatetimeIndex:
    """Issue-time cutoff (UTC) for each target period: D-1 ``hour:minute`` London time."""
    days = delivery_day(ensure_utc_index(target_times))
    lookup = {d: issue_time(d, hour, minute) for d in days.unique()}
    return pd.DatetimeIndex([lookup[d] for d in days], name="cutoff")


def select_vintage(
    forecasts: pd.DataFrame,
    *,
    hour: int = 10,
    minute: int = 0,
) -> pd.DataFrame:
    """Keep, for each target period, the latest vintage published by its issue-time cutoff.

    Args:
        forecasts: Long table with tz-aware UTC columns ``target_time`` and ``issued_at``,
            plus any number of value columns. One row per (target, vintage).
        hour: Issue hour (London time, on D-1).
        minute: Issue minute.

    Returns:
        Frame indexed by ``target_time`` with the value columns and the ``issued_at`` of the
        vintage chosen (kept for auditing). Targets with no vintage published in time are
        omitted, so callers should reindex onto their full period grid and treat gaps as missing.
    """
    missing = {TARGET, ISSUED} - set(forecasts.columns)
    if missing:
        raise KeyError(f"forecasts is missing columns: {sorted(missing)}")
    target = pd.DatetimeIndex(forecasts[TARGET])
    issued = pd.DatetimeIndex(forecasts[ISSUED])
    ensure_utc_index(target)
    ensure_utc_index(issued)

    eligible = forecasts[issued <= cutoff_times(target, hour, minute)]
    latest = eligible.sort_values([TARGET, ISSUED]).groupby(TARGET, sort=True).tail(1)
    return latest.set_index(TARGET)


def assert_available_at_issue(features: pd.DataFrame, *, hour: int = 10, minute: int = 0) -> None:
    """Raise if any row of ``features`` uses a vintage published after its cutoff.

    ``features`` must be indexed by target period (UTC) and carry an ``issued_at`` column, as
    returned by :func:`select_vintage`. Intended as a cheap runtime check in feature pipelines.
    """
    cutoffs = cutoff_times(features.index, hour, minute)
    late = pd.DatetimeIndex(features[ISSUED]) > cutoffs
    if late.any():
        first = features.index[late][0]
        raise AssertionError(f"{int(late.sum())} rows leak future data; first at {first}")
