"""Elexon Insights API loaders for NESO's demand (NDF) and wind (WINDFOR) forecast vintages.

Every published vintage is kept, so features can be built point-in-time with
:func:`gb_price_sandbox.features.vintage.select_vintage`. Output is a long table with one row
per (vintage, target period): ``issued_at`` (publish time, UTC), ``target_time`` (period start,
UTC) and the forecast value.

Publishing patterns (observed 2022-2026):

- **NDF** (national demand, MW, half-hourly): a rolling vintage every 30 minutes reaching to
  about 04:30 local time on the next day, plus one longer vintage at about 08:45 local on D-1
  covering the rest of D. Requests are limited to 1 day of publish times.
- **WINDFOR** (wind generation, MW, *hourly*): about 8 vintages a day, each covering about 48
  hours ahead. Requests are limited to 7 days of publish times. :func:`load_wind_forecast`
  interpolates each vintage to half-hourly on its own, so no value mixes two vintages.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.cache import cached_monthly
from gb_price_sandbox.data.http import JsonSession, get_json, make_session
from gb_price_sandbox.timing import PERIOD

NDF_MAX_SPAN = pd.Timedelta(days=1)
WINDFOR_MAX_SPAN = pd.Timedelta(days=7)

# Vintages published up to this long before the requested target window are loaded too, since
# a day-ahead forecast for day D is published on D-1 (or earlier).
LOOKBACK = pd.Timedelta(days=3)

TARGET = "target_time"
ISSUED = "issued_at"


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")


def _fetch_by_publish_time(
    dataset: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_span: pd.Timedelta,
    settings: Settings,
    session: JsonSession,
) -> list[dict]:
    """All rows of ``dataset`` published in ``[start, end)``, chunked to the API's limit."""
    url = f"{settings.elexon_base_url}/datasets/{dataset}"
    rows: list[dict] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + max_span, end)
        params = {
            "publishDateTimeFrom": _iso(cursor),
            # Both bounds are inclusive; stop a minute short so chunks don't overlap.
            "publishDateTimeTo": _iso(chunk_end - pd.Timedelta(minutes=1)),
        }
        rows.extend(get_json(session, url, params, settings.http_timeout)["data"])
        cursor = chunk_end
    return rows


def _tidy(
    rows: list[dict], value_key: str, value_col: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Rows to a frame indexed by ``issued_at`` with ``target_time`` and ``value_col``."""
    if not rows:
        empty = pd.DatetimeIndex([], tz="UTC", name=ISSUED)
        return pd.DataFrame(
            {TARGET: pd.DatetimeIndex([], tz="UTC"), value_col: pd.Series([], dtype=float)},
            index=empty,
        )
    raw = pd.DataFrame(rows)
    df = pd.DataFrame(
        {
            TARGET: pd.to_datetime(raw["startTime"], utc=True),
            value_col: raw[value_key].astype(float),
        }
    )
    df.index = pd.DatetimeIndex(pd.to_datetime(raw["publishTime"], utc=True), name=ISSUED)
    df = df[(df.index >= start) & (df.index < end)]
    keys = pd.MultiIndex.from_arrays([df.index, df[TARGET]])
    return df[~keys.duplicated(keep="last")].sort_index(kind="stable")


def fetch_demand_forecast_vintages(
    start: pd.Timestamp, end: pd.Timestamp, settings: Settings, session: JsonSession
) -> pd.DataFrame:
    """Pull every NDF vintage published in ``[start, end)`` (indexed by ``issued_at``)."""
    rows = _fetch_by_publish_time("NDF", start, end, NDF_MAX_SPAN, settings, session)
    rows = [r for r in rows if r.get("boundary", "N") == "N"]
    return _tidy(rows, "demand", "demand_mw", start, end)


def fetch_wind_forecast_vintages(
    start: pd.Timestamp, end: pd.Timestamp, settings: Settings, session: JsonSession
) -> pd.DataFrame:
    """Pull every hourly WINDFOR vintage published in ``[start, end)``, indexed by ``issued_at``."""
    rows = _fetch_by_publish_time("WINDFOR", start, end, WINDFOR_MAX_SPAN, settings, session)
    return _tidy(rows, "generation", "wind_mw", start, end)


def interpolate_half_hourly(vintages: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Add half-hour points between consecutive hourly targets *within each vintage*.

    The value at ``hh:30`` is the mean of ``hh:00`` and ``hh+1:00`` from the same vintage. A
    vintage's last hour gets no ``:30`` point (no extrapolation), and nothing is filled across
    gaps longer than an hour.

    Args:
        vintages: Long table with ``issued_at``, ``target_time`` and ``value_col`` columns.
        value_col: Column to interpolate.

    Returns:
        Long table with the same columns, sorted by ``issued_at`` then ``target_time``.
    """
    df = vintages.sort_values([ISSUED, TARGET]).reset_index(drop=True)
    nxt = df.groupby(ISSUED)[[TARGET, value_col]].shift(-1)
    adjacent = (nxt[TARGET] - df[TARGET]) == pd.Timedelta(hours=1)
    mids = pd.DataFrame(
        {
            ISSUED: df.loc[adjacent, ISSUED],
            TARGET: df.loc[adjacent, TARGET] + PERIOD,
            value_col: (df.loc[adjacent, value_col] + nxt.loc[adjacent, value_col]) / 2,
        }
    )
    out = pd.concat([df, mids], ignore_index=True)
    return out.sort_values([ISSUED, TARGET]).reset_index(drop=True)


def _load(
    name: str,
    fetch: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: Settings,
    refresh: bool,
) -> pd.DataFrame:
    raw = cached_monthly(
        settings.data_dir / "raw" / "elexon" / name,
        start - LOOKBACK,
        end,
        fetch,
        refresh=refresh,
    )
    out = raw.reset_index()
    in_window = (out[TARGET] >= start) & (out[TARGET] < end)
    return out.loc[in_window, [ISSUED, TARGET, *raw.columns.drop(TARGET)]].reset_index(drop=True)


def load_demand_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: Settings,
    *,
    session: JsonSession | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load all NDF vintages for target periods in ``[start, end)``, via the parquet cache.

    Args:
        start: Inclusive start of the target window (tz-aware).
        end: Exclusive end of the target window (tz-aware).
        settings: Application settings.
        session: HTTP session; one with retries is created if omitted.
        refresh: Re-fetch cached months.

    Returns:
        Long table with ``issued_at``, ``target_time`` and ``demand_mw``, one row per vintage
        and half-hour. Pass it to ``select_vintage`` to get point-in-time features.
    """
    session = session or make_session()
    return _load(
        "ndf",
        lambda a, b: fetch_demand_forecast_vintages(a, b, settings, session),
        start,
        end,
        settings,
        refresh,
    )


def load_wind_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: Settings,
    *,
    session: JsonSession | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load all WINDFOR vintages for target periods in ``[start, end)``, half-hourly.

    The raw cache stays hourly. Interpolation to half-hours happens after loading, separately
    within each vintage (see :func:`interpolate_half_hourly`).

    Args:
        start: Inclusive start of the target window (tz-aware).
        end: Exclusive end of the target window (tz-aware).
        settings: Application settings.
        session: HTTP session; one with retries is created if omitted.
        refresh: Re-fetch cached months.

    Returns:
        Long table with ``issued_at``, ``target_time`` and ``wind_mw``.
    """
    session = session or make_session()
    hourly = _load(
        "windfor",
        lambda a, b: fetch_wind_forecast_vintages(a, b, settings, session),
        # Widen by an hour so the last :30 point of the window can be interpolated.
        start,
        end + pd.Timedelta(hours=1),
        settings,
        refresh,
    )
    half_hourly = interpolate_half_hourly(hourly, "wind_mw")
    in_window = (half_hourly[TARGET] >= start) & (half_hourly[TARGET] < end)
    return half_hourly[in_window].reset_index(drop=True)
