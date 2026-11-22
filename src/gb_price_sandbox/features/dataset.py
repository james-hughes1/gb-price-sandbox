"""Model-ready half-hourly dataset: target price plus features available at issue time.

One row per settlement period (UTC start). Columns:

- ``price``: realised MID (the target). NaN where nothing traded or data is missing.
- ``demand_mw``, ``wind_mw``: point-in-time NESO forecasts (latest vintage published by the
  D-1 cutoff), and ``residual_demand_mw`` = demand - wind.
- ``price_lag_{k}d``: the price exactly ``k`` x 24 hours earlier. Only lags whose whole period
  has settled by the cutoff are allowed; with a D-1 morning cutoff that means k >= 2.
- ``price_mean_d2``: mean price over delivery day D-2.
- ``delivery_day``, ``period_of_day`` (local half-hour slot, 0-47), ``day_of_week`` (0 = Mon).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.cache import cached_parquet
from gb_price_sandbox.data.elexon import load_market_index_price
from gb_price_sandbox.data.elexon_forecasts import load_demand_forecast, load_wind_forecast
from gb_price_sandbox.features.vintage import cutoff_times, select_vintage
from gb_price_sandbox.timing import LONDON, PERIOD, delivery_day

#: 2, 3, 7: LEAR's lags. 14, 21, 28: with 7, the last four same weekdays (weekday profile).
PRICE_LAG_DAYS: tuple[int, ...] = (2, 3, 7, 14, 21, 28)
#: Bump when the dataset's columns change, so cached files are rebuilt.
DATASET_VERSION = 2
STUDY_START = dt.date(2022, 1, 1)

TARGET = "price"
#: The only columns models are allowed to see. Same-period realised data (``price``,
#: ``volume_mwh``) is deliberately absent.
FEATURE_COLUMNS: tuple[str, ...] = (
    "demand_mw",
    "wind_mw",
    "residual_demand_mw",
    *(f"price_lag_{k}d" for k in PRICE_LAG_DAYS),
    "price_mean_d2",
    "period_of_day",
    "day_of_week",
)


def london_midnight(day: dt.date) -> pd.Timestamp:
    """Start of GB delivery day ``day`` in UTC."""
    return pd.Timestamp(day).tz_localize(LONDON).tz_convert("UTC")


def check_lag_availability(
    index: pd.DatetimeIndex, lag_days: tuple[int, ...], hour: int = 10
) -> None:
    """Raise if any price lag would use a period that has not ended by the issue-time cutoff.

    Args:
        index: Target period starts (UTC).
        lag_days: Lags in days (each is exactly ``k`` x 24 hours).
        hour: Issue hour (London time, D-1).
    """
    cutoffs = cutoff_times(index, hour)
    for k in lag_days:
        source_end = index - pd.Timedelta(days=k) + PERIOD
        late = source_end > cutoffs
        if late.any():
            raise ValueError(
                f"price lag of {k} day(s) is not known by the D-1 {hour:02d}:00 cutoff "
                f"(first offending target: {index[late][0]})"
            )


def add_calendar_and_lag_features(
    frame: pd.DataFrame, lag_days: tuple[int, ...] = PRICE_LAG_DAYS, hour: int = 10
) -> pd.DataFrame:
    """Add calendar columns, price lags and the D-2 mean price to a frame with ``price``.

    ``frame`` must be indexed by a complete, regular half-hourly UTC grid, so that shifting
    by ``48 * k`` rows is exactly ``k`` days.
    """
    check_lag_availability(frame.index, lag_days, hour)
    out = frame.copy()
    local = out.index.tz_convert(LONDON)
    out["delivery_day"] = delivery_day(out.index)
    out["period_of_day"] = local.hour * 2 + local.minute // 30
    out["day_of_week"] = local.day_of_week
    for k in lag_days:
        out[f"price_lag_{k}d"] = out["price"].shift(48 * k)
    daily_mean = out.groupby("delivery_day")["price"].mean()
    shifted = pd.Series(daily_mean.to_numpy(), index=daily_mean.index + dt.timedelta(days=2))
    out["price_mean_d2"] = out["delivery_day"].map(shifted).astype(float)
    return out


def build_dataset(
    start_day: dt.date,
    end_day: dt.date,
    settings: Settings,
    *,
    hour: int = 10,
    refresh: bool = False,
) -> pd.DataFrame:
    """Assemble the model dataset for delivery days ``[start_day, end_day)``.

    The result is cached under ``data/processed``. Delete the file (or pass ``refresh``) to
    rebuild it. Lags reach back before ``start_day``, so the first week has NaN lags.

    Args:
        start_day: First delivery day (inclusive).
        end_day: Last delivery day (exclusive).
        settings: Application settings.
        hour: Forecast issue hour, London time on D-1.
        refresh: Rebuild even if a cached file exists.

    Returns:
        Half-hourly frame on a complete UTC grid, described in the module docstring.
    """
    path = (
        settings.data_dir
        / "processed"
        / f"dataset_v{DATASET_VERSION}_{start_day}_{end_day}_h{hour:02d}.parquet"
    )

    def build() -> pd.DataFrame:
        start, end = london_midnight(start_day), london_midnight(end_day)
        grid = pd.date_range(start, end, freq=PERIOD, inclusive="left", name="start_time")

        mid = load_market_index_price(start, end, settings)
        frame = pd.DataFrame(index=grid)
        frame["price"] = mid["price"].reindex(grid)
        frame["volume_mwh"] = mid["volume"].reindex(grid)

        for col, loader in (("demand_mw", load_demand_forecast), ("wind_mw", load_wind_forecast)):
            chosen = select_vintage(loader(start, end, settings), hour=hour)
            frame[col] = chosen[col].reindex(grid)
        frame["residual_demand_mw"] = frame["demand_mw"] - frame["wind_mw"]
        return add_calendar_and_lag_features(frame, PRICE_LAG_DAYS, hour)

    df = cached_parquet(path, build, refresh=refresh)
    df["delivery_day"] = pd.to_datetime(df["delivery_day"]).dt.date
    return df


def complete_days(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Delivery days on which every period has non-missing values in ``columns``."""
    ok = frame[columns].notna().all(axis=1).groupby(frame["delivery_day"]).all()
    return ok[ok].index.to_numpy()
