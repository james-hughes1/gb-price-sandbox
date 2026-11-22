"""Loader tests for Open-Meteo weather, PV_Live solar and GOV.UK bank holidays (offline)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.calendar import (
    bank_holiday_dates,
    is_christmas_period,
    load_bank_holidays,
)
from gb_price_sandbox.data.open_meteo import (
    SITES,
    Site,
    fetch_weather,
    load_weather,
    to_half_hourly,
)
from gb_price_sandbox.data.pv_live import fetch_solar_outturn, load_solar_outturn
from tests.fakes import FakeSession

UTC = "UTC"


# --- Open-Meteo ------------------------------------------------------------------------------


def _meteo_handler(url, params):
    """Hourly series per site: irradiance = hour of day x 10, temperature = 5 + site index."""
    start = pd.Timestamp(params["start_date"])
    end = pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23)
    times = pd.date_range(start, end, freq="h")
    n_sites = len(params["latitude"].split(","))
    return [
        {
            "hourly": {
                "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                "shortwave_radiation": [float(t.hour * 10) for t in times],
                "temperature_2m": [5.0 + i if t.hour != 3 else None for t in times],
            }
        }
        for i in range(n_sites)
    ]


def test_weather_irradiance_moved_to_hour_start_and_weighted():
    sites = (Site("A", 50, 0, 1.0, 1.0), Site("B", 51, 0, 3.0, 0.0))
    start = pd.Timestamp("2024-06-01", tz=UTC)
    end = pd.Timestamp("2024-06-02", tz=UTC)
    df = fetch_weather(start, end, Settings(), FakeSession(_meteo_handler), sites)
    assert df.index.min() == start and df.index.max() == end - pd.Timedelta(hours=1)
    # The value labelled 13:00 is the 12:00-13:00 mean, so it belongs to the 12:00 hour.
    assert df.loc[pd.Timestamp("2024-06-01 12:00", tz=UTC), "solar_irradiance_wm2"] == 130.0
    # Temperature weights 1:3 -> (5 * 1 + 6 * 3) / 4.
    assert df.loc[pd.Timestamp("2024-06-01 12:00", tz=UTC), "temperature_c"] == 5.75


def test_weather_weighted_mean_skips_missing_sites():
    sites = (Site("A", 50, 0, 1.0, 1.0), Site("B", 51, 0, 1.0, 1.0))

    def handler(url, params):
        payload = _meteo_handler(url, params)
        payload[1]["hourly"]["temperature_2m"] = [None] * len(payload[1]["hourly"]["time"])
        return payload

    start = pd.Timestamp("2024-06-01", tz=UTC)
    df = fetch_weather(start, start + pd.Timedelta(days=1), Settings(), FakeSession(handler), sites)
    assert df.loc[pd.Timestamp("2024-06-01 12:00", tz=UTC), "temperature_c"] == 5.0
    assert np.isnan(df.loc[pd.Timestamp("2024-06-01 03:00", tz=UTC), "temperature_c"])


def test_to_half_hourly():
    hours = pd.date_range("2024-06-01", periods=4, freq="h", tz=UTC)
    hourly = pd.DataFrame(
        {"solar_irradiance_wm2": [0.0, 100.0, 200.0, 300.0], "temperature_c": [10, 12, np.nan, 16]},
        index=hours,
    )
    grid = pd.date_range(hours[0], periods=8, freq="30min", tz=UTC)
    out = to_half_hourly(hourly, grid)
    assert list(out["solar_irradiance_wm2"]) == [0, 0, 100, 100, 200, 200, 300, 300]
    assert out["temperature_c"].iloc[1] == 11.0  # midway between 10 and 12
    assert np.isnan(out["temperature_c"].iloc[4])  # the missing hour is not bridged
    assert np.isnan(out["temperature_c"].iloc[7])  # nothing after the last hour


def test_load_weather_caches_and_returns_half_hourly(tmp_path):
    settings = Settings(data_dir=tmp_path)
    session = FakeSession(_meteo_handler)
    start = pd.Timestamp("2024-01-01", tz=UTC)
    end = pd.Timestamp("2024-03-01", tz=UTC)
    first = load_weather(start, end, settings, session=session)
    calls = len(session.calls)
    again = load_weather(start, end, settings, session=session)
    assert len(session.calls) == calls  # served from cache
    pd.testing.assert_frame_equal(first, again)
    assert len(first) == 60 * 48
    assert first["solar_irradiance_wm2"].notna().all()


def test_site_weights_sum_to_one():
    assert sum(s.temperature_weight for s in SITES) == pytest.approx(1.0)
    assert sum(s.solar_weight for s in SITES) == pytest.approx(1.0)


# --- PV_Live ---------------------------------------------------------------------------------


def _pv_handler(url, params):
    """Rows labelled by period END, newest first, with inclusive bounds (like the real API)."""
    lo = pd.Timestamp(params["start"], tz=UTC)
    hi = pd.Timestamp(params["end"], tz=UTC)
    ends = pd.date_range(lo, hi, freq="30min")[::-1]
    return {
        "meta": ["gsp_id", "datetime_gmt", "generation_mw"],
        "data": [[0, t.strftime("%Y-%m-%dT%H:%M:%SZ"), float(t.hour)] for t in ends],
    }


def test_pv_live_relabels_to_period_start():
    start = pd.Timestamp("2024-06-01", tz=UTC)
    end = pd.Timestamp("2024-06-02", tz=UTC)
    df = fetch_solar_outturn(start, end, Settings(), FakeSession(_pv_handler))
    assert len(df) == 48
    assert df.index.min() == start and df.index.max() == end - pd.Timedelta(minutes=30)
    assert df.index.is_monotonic_increasing
    # Period 11:30-12:00 was labelled 12:00 by the API.
    assert df.loc[pd.Timestamp("2024-06-01 11:30", tz=UTC), "solar_mw"] == 12.0


def test_pv_live_cache_idempotent(tmp_path):
    settings = Settings(data_dir=tmp_path)
    session = FakeSession(_pv_handler)
    start = pd.Timestamp("2024-01-01", tz=UTC)
    end = pd.Timestamp("2024-02-01", tz=UTC)
    first = load_solar_outturn(start, end, settings, session=session)
    calls = len(session.calls)
    again = load_solar_outturn(start, end, settings, session=session)
    assert len(session.calls) == calls
    pd.testing.assert_frame_equal(first, again)


# --- bank holidays ---------------------------------------------------------------------------


def _holidays_handler(last_year):
    def handler(url, params):
        events = [
            {"title": "Christmas Day", "date": f"{y}-12-25"} for y in range(2019, last_year + 1)
        ]
        return {"england-and-wales": {"division": "england-and-wales", "events": events}}

    return handler


def test_bank_holidays_cached_and_refetched_when_stale(tmp_path):
    settings = Settings(data_dir=tmp_path)
    session = FakeSession(_holidays_handler(2026))
    df = load_bank_holidays(settings, session=session)
    assert dt.date(2025, 12, 25) in bank_holiday_dates(df)
    load_bank_holidays(settings, until=dt.date(2026, 6, 1), session=session)
    assert len(session.calls) == 1  # cache covers the request

    session.handler = _holidays_handler(2028)
    df = load_bank_holidays(settings, until=dt.date(2027, 6, 1), session=session)
    assert len(session.calls) == 2  # cache ended in 2026, so it was refreshed
    assert dt.date(2028, 12, 25) in bank_holiday_dates(df)


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (dt.date(2024, 12, 23), False),
        (dt.date(2024, 12, 24), True),
        (dt.date(2024, 12, 31), True),
        (dt.date(2025, 1, 1), True),
        (dt.date(2025, 1, 2), False),
    ],
)
def test_christmas_period(day, expected):
    assert is_christmas_period(day) is expected
