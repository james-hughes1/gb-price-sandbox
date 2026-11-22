"""Open-Meteo historical weather forecasts: GB-weighted solar irradiance and temperature.

Open-Meteo's *historical forecast* API archives the first hours of every weather-model run and
stitches them together. So each value is a short-lead forecast (a few hours ahead), not the
forecast a trader would have seen at 10:00 on D-1. That makes these series optimistic: fine for
exploring how weather relates to price, but a backtest would need the issue-time vintage (Open-
Meteo's *previous runs* API, which only goes back to early 2024).

Point forecasts at a handful of sites are combined into GB-wide series with fixed weights:
roughly population for temperature (it drives heating demand) and roughly installed solar
capacity by region for irradiance. Both weightings are approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.cache import cached_monthly
from gb_price_sandbox.data.http import JsonSession, get_json, make_session
from gb_price_sandbox.timing import PERIOD


@dataclass(frozen=True)
class Site:
    """A forecast location and its weights in the GB averages."""

    name: str
    lat: float
    lon: float
    temperature_weight: float
    solar_weight: float


SITES: tuple[Site, ...] = (
    Site("London", 51.51, -0.13, 0.30, 0.10),
    Site("Birmingham", 52.49, -1.89, 0.15, 0.15),
    Site("Manchester", 53.48, -2.24, 0.15, 0.05),
    Site("Leeds", 53.80, -1.55, 0.10, 0.05),
    Site("Bristol", 51.45, -2.59, 0.10, 0.10),
    Site("Glasgow", 55.86, -4.25, 0.10, 0.05),
    Site("Newcastle", 54.98, -1.61, 0.05, 0.0),
    Site("Cardiff", 51.48, -3.18, 0.05, 0.05),
    Site("Exeter", 50.72, -3.53, 0.0, 0.20),
    Site("Southampton", 50.90, -1.40, 0.0, 0.15),
    Site("Cambridge", 52.21, 0.12, 0.0, 0.10),
)

HOURLY_VARS = ("shortwave_radiation", "temperature_2m")
COLUMNS = ("solar_irradiance_wm2", "temperature_c")

_EMPTY_INDEX = pd.DatetimeIndex([], tz="UTC", name="start_time")


def _weighted(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean across sites (columns), renormalising over the sites with data."""
    present = ~np.isnan(values)
    w = np.where(present, weights[None, :], 0.0)
    total = w.sum(axis=1)
    out = np.where(present, values, 0.0) @ weights / np.where(total > 0, total, 1.0)
    return np.where(total > 0, out, np.nan)


def fetch_weather(
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: Settings,
    session: JsonSession,
    sites: tuple[Site, ...] = SITES,
) -> pd.DataFrame:
    """Pull hourly GB-weighted irradiance and temperature for ``[start, end)`` in one request.

    Open-Meteo reports ``shortwave_radiation`` as the mean over the *preceding* hour, so it is
    moved back an hour to label each value by the start of the hour it covers. Temperature is
    instantaneous and keeps its timestamp.

    Returns:
        Hourly frame (UTC hour starts) with ``solar_irradiance_wm2`` and ``temperature_c``.
    """
    params = {
        "latitude": ",".join(f"{s.lat}" for s in sites),
        "longitude": ",".join(f"{s.lon}" for s in sites),
        "start_date": f"{start.tz_convert('UTC'):%Y-%m-%d}",
        # One extra day so the last hour's (backward-averaged) irradiance is included.
        "end_date": f"{end.tz_convert('UTC'):%Y-%m-%d}",
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "GMT",
    }
    payload = get_json(session, f"{settings.open_meteo_base_url}/forecast", params, 60.0)
    if isinstance(payload, dict):  # a single site comes back as an object, not a list
        payload = [payload]
    times = pd.DatetimeIndex(pd.to_datetime(payload[0]["hourly"]["time"], utc=True))

    def stack(var: str) -> np.ndarray:
        # dtype=float turns the API's nulls into NaN.
        return np.column_stack([np.asarray(p["hourly"][var], dtype=float) for p in payload])

    irr = _weighted(stack("shortwave_radiation"), np.array([s.solar_weight for s in sites]))
    temp = _weighted(stack("temperature_2m"), np.array([s.temperature_weight for s in sites]))
    df = pd.DataFrame({"temperature_c": temp}, index=pd.DatetimeIndex(times, name="start_time"))
    df["solar_irradiance_wm2"] = pd.Series(irr, index=times - pd.Timedelta(hours=1)).reindex(
        df.index
    )
    df = df[list(COLUMNS)]
    return df[(df.index >= start) & (df.index < end)]


def to_half_hourly(hourly: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Put hourly weather on a half-hourly grid.

    Irradiance is an hourly mean, so both half-hours of an hour get the same value.
    Temperature is instantaneous: on-the-hour periods take it directly and half-past periods
    take the average of the hours either side (never bridging a missing hour).
    """
    out = pd.DataFrame(index=grid)
    out["solar_irradiance_wm2"] = hourly["solar_irradiance_wm2"].reindex(grid.floor("h")).to_numpy()
    temp = hourly["temperature_c"].reindex(grid)
    out["temperature_c"] = temp.interpolate(limit=1, limit_area="inside")
    return out


def load_weather(
    start: pd.Timestamp,
    end: pd.Timestamp,
    settings: Settings,
    *,
    session: JsonSession | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load GB-weighted irradiance and temperature forecasts, half-hourly, via the cache.

    Args:
        start: Inclusive start (tz-aware).
        end: Exclusive end (tz-aware).
        settings: Application settings.
        session: HTTP session; one with retries is created if omitted.
        refresh: Re-fetch cached months.

    Returns:
        Frame on the half-hourly UTC grid ``[start, end)`` with ``solar_irradiance_wm2`` (W/m²,
        mean over the hour) and ``temperature_c`` (°C).
    """
    session = session or make_session()
    hourly = cached_monthly(
        settings.data_dir / "raw" / "open_meteo" / "gb_weighted",
        start,
        end + pd.Timedelta(hours=1),  # the next hour, to interpolate the last half-hour
        lambda a, b: fetch_weather(a, b, settings, session),
        refresh=refresh,
    )
    grid = pd.date_range(start, end, freq=PERIOD, inclusive="left", name="start_time")
    return to_half_hourly(hourly, grid)
