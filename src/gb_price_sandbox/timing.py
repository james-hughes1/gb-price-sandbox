"""Time conventions: UTC storage, Europe/London delivery days, forecast issue times.

All timestamps in the package are tz-aware UTC and mark the *start* of a half-hourly
settlement period. GB delivery days are local (Europe/London) calendar days, so they have
48 periods normally, 46 on the spring clock change and 50 on the autumn one.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

LONDON = "Europe/London"
PERIOD = pd.Timedelta(minutes=30)
PERIOD_HOURS = 0.5


def ensure_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    """Return ``index`` as a tz-aware UTC DatetimeIndex, rejecting naive timestamps."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("expected a DatetimeIndex")
    if index.tz is None:
        raise ValueError("timestamps must be tz-aware (UTC); naive timestamps are ambiguous")
    return index.tz_convert("UTC")


def delivery_day(index: pd.Index) -> pd.Index:
    """Map each period start to its GB delivery day (local calendar date)."""
    return pd.Index(ensure_utc_index(index).tz_convert(LONDON).date, name="delivery_day")


def issue_time(day: dt.date, hour: int = 10, minute: int = 0) -> pd.Timestamp:
    """Forecast issue time for delivery day ``day``: D-1 at ``hour:minute`` London time, in UTC.

    Every feature used to forecast day ``day`` must have been published at or before this
    time. The default of 10:00 is a modelling assumption: a forecaster trading the GB
    day-ahead auctions on the morning of D-1.
    """
    local = pd.Timestamp(day - dt.timedelta(days=1)).replace(hour=hour, minute=minute)
    return local.tz_localize(LONDON).tz_convert("UTC")


def day_periods(day: dt.date) -> pd.DatetimeIndex:
    """All half-hourly period starts (UTC) for GB delivery day ``day`` (46, 48 or 50)."""
    start = pd.Timestamp(day).tz_localize(LONDON)
    end = pd.Timestamp(day + dt.timedelta(days=1)).tz_localize(LONDON)
    return pd.date_range(start, end, freq=PERIOD, inclusive="left").tz_convert("UTC")
