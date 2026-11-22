"""Sheffield Solar PV_Live: estimated GB solar generation, half-hourly.

PV_Live upscales metered output from a sample of PV systems to an estimate for the whole GB
fleet (region 0 = national). Almost all of it is *embedded*: connected to local distribution
networks rather than the transmission grid, so NESO sees it only as lower demand.

These are outturn estimates, revised over the following day, so they are **not** known at
forecast issue time for the day being forecast. Lags of two days or more are safe.
"""

from __future__ import annotations

import pandas as pd

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.cache import cached_monthly
from gb_price_sandbox.data.http import JsonSession, get_json, make_session
from gb_price_sandbox.timing import PERIOD

NATIONAL = 0

_EMPTY_INDEX = pd.DatetimeIndex([], tz="UTC", name="start_time")


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S")


def fetch_solar_outturn(
    start: pd.Timestamp, end: pd.Timestamp, settings: Settings, session: JsonSession
) -> pd.DataFrame:
    """Pull national PV_Live generation for periods starting in ``[start, end)``.

    PV_Live labels each half-hour by its *end* time (and both ``start`` and ``end`` query
    bounds are inclusive), so timestamps are moved back 30 minutes to the package's
    period-start convention.
    """
    params = {"start": _iso(start + PERIOD), "end": _iso(end)}
    url = f"{settings.pv_live_base_url}/gsp/{NATIONAL}"
    payload = get_json(session, url, params, settings.http_timeout)
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame({"solar_mw": []}, index=_EMPTY_INDEX, dtype=float)
    meta = payload.get("meta", ["gsp_id", "datetime_gmt", "generation_mw"])
    df = pd.DataFrame(rows, columns=meta)
    ends = pd.DatetimeIndex(pd.to_datetime(df["datetime_gmt"], utc=True))
    out = pd.DataFrame(
        {"solar_mw": df["generation_mw"].astype(float).to_numpy()},
        index=pd.DatetimeIndex(ends - PERIOD, name="start_time"),
    )
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[(out.index >= start) & (out.index < end)]


def load_solar_outturn(
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: Settings,
    *,
    session: JsonSession | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load national PV_Live solar generation estimates (MW), via the parquet cache.

    Args:
        start: Inclusive start (tz-aware).
        end: Exclusive end (tz-aware).
        settings: Application settings.
        session: HTTP session; one with retries is created if omitted.
        refresh: Re-fetch cached months.

    Returns:
        Frame indexed by period start (UTC) with ``solar_mw``.
    """
    session = session or make_session()
    return cached_monthly(
        settings.data_dir / "raw" / "pv_live" / "national",
        start,
        end,
        lambda a, b: fetch_solar_outturn(a, b, settings, session),
        refresh=refresh,
    )
