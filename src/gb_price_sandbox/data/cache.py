"""Idempotent local parquet cache for raw API pulls."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

RangeFetcher = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]

# Months ending more recently than this are re-fetched rather than cached, because the
# source may still be publishing or revising data for them.
SETTLE_MARGIN = pd.Timedelta(days=7)


def cached_parquet(
    path: Path,
    fetch: Callable[[], pd.DataFrame],
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the frame cached at ``path``, calling ``fetch`` only if it is missing.

    Writes go to a temporary file first and are then renamed, so an interrupted
    pull never leaves a half-written cache file behind.

    Args:
        path: Parquet file location.
        fetch: Zero-argument callable that pulls the data from source.
        refresh: If True, ignore any existing cache and re-fetch.

    Returns:
        The cached (or freshly fetched) DataFrame.
    """
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)
    return df


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """UTC month starts of every calendar month overlapping ``[start, end)``."""
    first = start.tz_convert("UTC").tz_localize(None).to_period("M").to_timestamp()
    months = pd.date_range(first, end.tz_convert("UTC").tz_localize(None), freq="MS")
    return [m.tz_localize("UTC") for m in months if m.tz_localize("UTC") < end]


def cached_monthly(
    directory: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch_range: RangeFetcher,
    *,
    now: pd.Timestamp | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch ``[start, end)`` through a cache partitioned into one parquet file per UTC month.

    Whole months are always fetched and cached, so the cache contents never depend on the
    range requested. Recent months (ending within :data:`SETTLE_MARGIN` of ``now``) are
    fetched but not written, so they are picked up again once complete.

    Args:
        directory: Folder holding ``YYYY-MM.parquet`` files.
        start: Inclusive start (tz-aware).
        end: Exclusive end (tz-aware).
        fetch_range: Callable pulling ``[a, b)`` from source and returning a frame indexed by
            tz-aware UTC timestamps.
        now: Current time, injectable for tests. Defaults to the wall clock.
        refresh: Re-fetch and overwrite cached months.

    Returns:
        Concatenated frame, sorted, restricted to ``[start, end)``.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    parts = []
    for month in month_starts(start, end):
        month_end = month + pd.offsets.MonthBegin(1)
        if month_end <= now - SETTLE_MARGIN:
            path = directory / f"{month:%Y-%m}.parquet"
            parts.append(
                cached_parquet(
                    path, lambda m=month, e=month_end: fetch_range(m, e), refresh=refresh
                )
            )
        else:
            parts.append(fetch_range(month, month_end))
    df = pd.concat(parts).sort_index()
    return df[(df.index >= start) & (df.index < end)]
