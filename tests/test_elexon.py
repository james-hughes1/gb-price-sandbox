import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.elexon import (
    fetch_market_index_price,
    fetch_system_price,
    load_market_index_price,
    load_system_price,
)
from gb_price_sandbox.data.http import make_session
from gb_price_sandbox.timing import PERIOD, day_periods
from tests.fakes import FakeSession

UTC = "UTC"


def _mid_handler(url, params):
    """Fake MID endpoint: one APX row per half-hour in [from, to] inclusive, plus N2EX noise.

    Every period at 03:00 UTC has zero volume.
    """
    lo = pd.Timestamp(params["from"])
    hi = pd.Timestamp(params["to"])
    assert hi - lo <= pd.Timedelta(days=7), "request exceeds the API's 7-day limit"
    rows = []
    for ts in pd.date_range(lo, hi, freq="30min"):
        zero = ts.hour == 3 and ts.minute == 0
        rows.append(
            {
                "startTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dataProvider": "APXMIDP",
                "price": 0.0 if zero else 50.0 + ts.hour,
                "volume": 0.0 if zero else 500.0,
            }
        )
        rows.append(
            {
                "startTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dataProvider": "N2EXMIDP",
                "price": 0.0,
                "volume": 0.0,
            }
        )
    return {"data": rows}


def _sp_handler(url, params):
    day = dt.date.fromisoformat(url.rsplit("/", 1)[1])
    return {
        "data": [
            {
                "startTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "systemSellPrice": 80.0,
                "systemBuyPrice": 80.0,
                "netImbalanceVolume": -100.0,
            }
            for ts in day_periods(day)
        ]
    }


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path)


def test_mid_chunks_requests_and_covers_range(settings):
    session = FakeSession(_mid_handler)
    start = pd.Timestamp("2024-01-01", tz=UTC)
    end = pd.Timestamp("2024-01-20", tz=UTC)

    df = fetch_market_index_price(start, end, settings, session)

    assert len(session.calls) == 3  # 7 + 7 + 5 days
    expected = pd.date_range(start, end, freq=PERIOD, inclusive="left")
    pd.testing.assert_index_equal(df.index, expected, check_names=False, check_freq=False)
    assert str(df.index.tz) == UTC


def test_mid_zero_volume_is_nan_and_n2ex_dropped(settings):
    session = FakeSession(_mid_handler)
    df = fetch_market_index_price(
        pd.Timestamp("2024-01-01", tz=UTC), pd.Timestamp("2024-01-02", tz=UTC), settings, session
    )
    assert len(df) == 48
    assert np.isnan(df.loc[pd.Timestamp("2024-01-01 03:00", tz=UTC), "price"])
    assert df["price"].notna().sum() == 47


def test_system_price_one_request_per_local_day(settings):
    session = FakeSession(_sp_handler)
    # One BST day, expressed in UTC: 2024-05-31 23:00 -> 2024-06-01 23:00.
    start = pd.Timestamp("2024-05-31 23:00", tz=UTC)
    end = pd.Timestamp("2024-06-01 23:00", tz=UTC)

    df = fetch_system_price(start, end, settings, session)

    assert [u.rsplit("/", 1)[1] for u, _ in session.calls] == ["2024-06-01"]
    assert len(df) == 48
    assert list(df.columns) == ["system_price", "net_imbalance_volume"]


def test_load_mid_uses_cache_for_complete_months(settings):
    session = FakeSession(_mid_handler)
    start = pd.Timestamp("2024-01-10", tz=UTC)
    end = pd.Timestamp("2024-02-05", tz=UTC)

    first = load_market_index_price(start, end, settings, session=session)
    n_calls = len(session.calls)
    second = load_market_index_price(start, end, settings, session=session)

    assert len(session.calls) == n_calls  # second load served entirely from parquet
    pd.testing.assert_frame_equal(first, second)
    assert sorted(p.name for p in (settings.data_dir / "raw/elexon/mid").iterdir()) == [
        "2024-01.parquet",
        "2024-02.parquet",
    ]
    assert first.index.min() == start
    assert first.index.max() == end - PERIOD


def test_load_system_price_returns_requested_range(settings):
    session = FakeSession(_sp_handler)
    start = pd.Timestamp("2024-03-30", tz=UTC)
    end = pd.Timestamp("2024-04-02", tz=UTC)
    df = load_system_price(start, end, settings, session=session)
    expected = pd.date_range(start, end, freq=PERIOD, inclusive="left")
    pd.testing.assert_index_equal(df.index, expected, check_names=False, check_freq=False)


@pytest.mark.network
def test_live_mid_known_values(settings):
    """Spot-check against values read off the live API (2024-01-01, periods 1-2)."""
    df = fetch_market_index_price(
        pd.Timestamp("2024-01-01 00:00", tz=UTC),
        pd.Timestamp("2024-01-01 01:00", tz=UTC),
        settings,
        make_session(),
    )
    assert df["price"].tolist() == pytest.approx([36.51, 45.17])
