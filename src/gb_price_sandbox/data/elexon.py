"""Elexon Insights API loaders: Market Index Price (MID) and system price.

No API key is needed. Both series are half-hourly, indexed by settlement-period start in UTC.
"""

from __future__ import annotations

import pandas as pd

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.cache import cached_monthly
from gb_price_sandbox.data.http import JsonSession, get_json, make_session
from gb_price_sandbox.timing import LONDON, PERIOD

MID_PROVIDER = "APXMIDP"  # The N2EX provider series carries zero volume throughout.
MID_MAX_SPAN = pd.Timedelta(days=7)  # API limit; ``from`` and ``to`` are both inclusive.

_EMPTY_INDEX = pd.DatetimeIndex([], tz="UTC", name="start_time")


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")


def fetch_market_index_price(
    start: pd.Timestamp, end: pd.Timestamp, settings: Settings, session: JsonSession
) -> pd.DataFrame:
    """Pull MID for ``[start, end)`` straight from the API, in chunks of at most 7 days.

    Periods with zero traded volume have no meaningful price (the API reports 0), so their
    price is set to NaN.
    """
    url = f"{settings.elexon_base_url}/balancing/pricing/market-index"
    rows: list[dict] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + MID_MAX_SPAN, end)
        params = {
            "from": _iso(cursor),
            "to": _iso(chunk_end - PERIOD),
            "dataProviders": MID_PROVIDER,
        }
        rows.extend(get_json(session, url, params, settings.http_timeout)["data"])
        cursor = chunk_end

    if not rows:
        return pd.DataFrame({"price": [], "volume": []}, index=_EMPTY_INDEX, dtype=float)
    df = pd.DataFrame(rows)
    df = df[df["dataProvider"] == MID_PROVIDER]
    df.index = pd.DatetimeIndex(pd.to_datetime(df["startTime"], utc=True), name="start_time")
    df = df[["price", "volume"]].astype(float)
    df.loc[df["volume"] <= 0, "price"] = float("nan")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df.index >= start) & (df.index < end)]


def fetch_system_price(
    start: pd.Timestamp, end: pd.Timestamp, settings: Settings, session: JsonSession
) -> pd.DataFrame:
    """Pull system (imbalance) prices for ``[start, end)``, one request per settlement day.

    Since the move to a single imbalance price (BSC modification P305, November 2015) system
    buy and sell prices are identical, so only the sell price is kept.
    """
    first = start.tz_convert(LONDON).date()
    last = (end - PERIOD).tz_convert(LONDON).date()
    rows: list[dict] = []
    for day in pd.date_range(first, last, freq="D"):
        url = f"{settings.elexon_base_url}/balancing/settlement/system-prices/{day:%Y-%m-%d}"
        rows.extend(get_json(session, url, None, settings.http_timeout)["data"])

    if not rows:
        return pd.DataFrame(
            {"system_price": [], "net_imbalance_volume": []}, index=_EMPTY_INDEX, dtype=float
        )
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(pd.to_datetime(df["startTime"], utc=True), name="start_time")
    df = df.rename(
        columns={"systemSellPrice": "system_price", "netImbalanceVolume": "net_imbalance_volume"}
    )[["system_price", "net_imbalance_volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df.index >= start) & (df.index < end)]


def load_market_index_price(
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: Settings,
    *,
    session: JsonSession | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load half-hourly Market Index Price (day-ahead price proxy), via the parquet cache.

    Args:
        start: Inclusive start of the delivery period (tz-aware).
        end: Exclusive end of the delivery period (tz-aware).
        settings: Application settings (base URL, cache dir, timeout).
        session: HTTP session; one with retries is created if omitted.
        refresh: Re-fetch cached months.

    Returns:
        DataFrame indexed by settlement-period start (UTC) with ``price`` (GBP/MWh, NaN when
        nothing traded) and ``volume`` (MWh) columns.
    """
    session = session or make_session()
    return cached_monthly(
        settings.data_dir / "raw" / "elexon" / "mid",
        start,
        end,
        lambda a, b: fetch_market_index_price(a, b, settings, session),
        refresh=refresh,
    )


def load_system_price(
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: Settings,
    *,
    session: JsonSession | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load half-hourly system (imbalance) price, via the parquet cache.

    Args:
        start: Inclusive start of the delivery period (tz-aware).
        end: Exclusive end of the delivery period (tz-aware).
        settings: Application settings.
        session: HTTP session; one with retries is created if omitted.
        refresh: Re-fetch cached months.

    Returns:
        DataFrame indexed by settlement-period start (UTC) with ``system_price`` (GBP/MWh) and
        ``net_imbalance_volume`` (MWh, positive = system short) columns.
    """
    session = session or make_session()
    return cached_monthly(
        settings.data_dir / "raw" / "elexon" / "system_price",
        start,
        end,
        lambda a, b: fetch_system_price(a, b, settings, session),
        refresh=refresh,
    )
