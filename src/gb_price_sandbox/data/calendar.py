"""GB calendar: England and Wales bank holidays from GOV.UK.

England and Wales are used for the whole of GB because they hold most of the demand. Scotland
has a few different holidays (2 January, St Andrew's Day, an August holiday on a different
Monday), which this ignores. The GOV.UK feed covers 2019 to roughly two years ahead.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.cache import cached_parquet
from gb_price_sandbox.data.http import JsonSession, get_json, make_session

DIVISION = "england-and-wales"


def fetch_bank_holidays(settings: Settings, session: JsonSession) -> pd.DataFrame:
    """Pull every England and Wales bank holiday in the GOV.UK feed.

    Returns:
        Frame with columns ``date`` (datetime64, midnight) and ``title``, sorted by date.
    """
    payload = get_json(session, settings.bank_holidays_url, None, settings.http_timeout)
    events = payload[DIVISION]["events"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([e["date"] for e in events]),
            "title": [e["title"] for e in events],
        }
    )
    return df.sort_values("date").reset_index(drop=True)


def load_bank_holidays(
    settings: Settings,
    *,
    until: dt.date | None = None,
    session: JsonSession | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Bank holidays via the parquet cache, re-fetched if the cache stops before ``until``.

    Args:
        settings: Application settings.
        until: Last date the caller needs covered. If the cached feed ends earlier (it only
            lists a couple of years ahead), the feed is pulled again.
        session: HTTP session; one with retries is created if omitted.
        refresh: Re-fetch even if cached.

    Returns:
        Frame with ``date`` and ``title`` columns.
    """
    path = settings.data_dir / "raw" / "calendar" / "bank_holidays_england_wales.parquet"
    session = session or make_session()

    def fetch() -> pd.DataFrame:
        return fetch_bank_holidays(settings, session)

    df = cached_parquet(path, fetch, refresh=refresh)
    if until is not None and not refresh and df["date"].max().date() < until:
        df = cached_parquet(path, fetch, refresh=True)
    return df


def bank_holiday_dates(holidays: pd.DataFrame) -> set[dt.date]:
    """The set of bank holiday dates in a :func:`load_bank_holidays` frame."""
    return set(pd.to_datetime(holidays["date"]).dt.date)


def is_christmas_period(day: dt.date) -> bool:
    """True from 24 December to 1 January inclusive, when demand and trading are unusual."""
    return (day.month == 12 and day.day >= 24) or (day.month == 1 and day.day == 1)
