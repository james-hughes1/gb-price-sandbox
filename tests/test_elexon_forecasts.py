import datetime as dt

import pandas as pd
import pytest

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.elexon_forecasts import (
    fetch_demand_forecast_vintages,
    fetch_wind_forecast_vintages,
    interpolate_half_hourly,
    load_demand_forecast,
    load_wind_forecast,
)
from gb_price_sandbox.data.http import make_session
from gb_price_sandbox.features.vintage import assert_available_at_issue, select_vintage
from gb_price_sandbox.timing import day_periods, issue_time
from tests.fakes import FakeSession

UTC = "UTC"


def _ts(s):
    return pd.Timestamp(s, tz=UTC)


def _publishes(params, every, offset):
    lo = pd.Timestamp(params["publishDateTimeFrom"])
    hi = pd.Timestamp(params["publishDateTimeTo"])
    first = lo.floor(every) + offset
    return [p for p in pd.date_range(first - pd.Timedelta(every), hi, freq=every) if lo <= p <= hi]


def ndf_handler(url, params):
    """Fake NDF: rolling vintages at :15/:45 reaching 03:30 UTC tomorrow, and a 07:45 UTC
    day-ahead vintage covering 04:00 tomorrow to 03:30 the day after (as observed in 2023)."""
    lo = pd.Timestamp(params["publishDateTimeFrom"])
    hi = pd.Timestamp(params["publishDateTimeTo"])
    assert hi - lo <= pd.Timedelta(days=1), "NDF request exceeds 1 day"
    rows = []
    for pub in _publishes(params, "30min", pd.Timedelta(minutes=15)):
        tomorrow = pub.normalize() + pd.Timedelta(days=1)
        if pub.hour == 7 and pub.minute == 45:
            targets = pd.date_range(tomorrow + pd.Timedelta(hours=4), periods=48, freq="30min")
        else:
            targets = pd.date_range(
                pub + pd.Timedelta(minutes=15), tomorrow + pd.Timedelta(hours=3.5), freq="30min"
            )
        for t in targets:
            rows.append(
                {
                    "publishTime": pub.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "startTime": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "demand": 30000.0 + pub.hour,  # encodes the vintage, to check selection
                    "boundary": "N",
                }
            )
    return {"data": rows}


def windfor_handler(url, params):
    """Fake WINDFOR: hourly vintages every 3 hours, each covering 20:00 today + 48 h."""
    lo = pd.Timestamp(params["publishDateTimeFrom"])
    hi = pd.Timestamp(params["publishDateTimeTo"])
    assert hi - lo <= pd.Timedelta(days=7), "WINDFOR request exceeds 7 days"
    rows = []
    for pub in _publishes(params, "3h", pd.Timedelta(minutes=30)):
        first = pub.normalize() + pd.Timedelta(hours=20)
        for k, t in enumerate(pd.date_range(first, periods=49, freq="1h")):
            rows.append(
                {
                    "publishTime": pub.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "startTime": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "generation": 1000.0 * pub.hour + 10.0 * k,
                }
            )
    return {"data": rows}


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path)


def test_ndf_requests_respect_one_day_limit(settings):
    session = FakeSession(ndf_handler)
    df = fetch_demand_forecast_vintages(_ts("2023-06-01"), _ts("2023-06-04"), settings, session)
    assert len(session.calls) == 3
    assert df.index.min() >= _ts("2023-06-01") and df.index.max() < _ts("2023-06-04")
    assert not df.reset_index().duplicated(["issued_at", "target_time"]).any()


def test_windfor_requests_respect_seven_day_limit(settings):
    session = FakeSession(windfor_handler)
    fetch_wind_forecast_vintages(_ts("2023-06-01"), _ts("2023-06-20"), settings, session)
    assert len(session.calls) == 3


def test_interpolation_stays_within_vintage():
    v = pd.DataFrame(
        {
            "issued_at": [_ts("2023-06-01 08:30")] * 3 + [_ts("2023-06-01 10:30")] * 2,
            "target_time": [
                _ts("2023-06-02 00:00"),
                _ts("2023-06-02 01:00"),
                _ts("2023-06-02 02:00"),
                _ts("2023-06-02 02:00"),
                _ts("2023-06-02 03:00"),
            ],
            "wind_mw": [100.0, 200.0, 400.0, 1000.0, 2000.0],
        }
    )
    out = interpolate_half_hourly(v, "wind_mw")
    first = out[out["issued_at"] == _ts("2023-06-01 08:30")]
    assert first["wind_mw"].tolist() == [100.0, 150.0, 200.0, 300.0, 400.0]
    # The 02:00-03:00 midpoint of the later vintage uses only its own values.
    later = out[out["issued_at"] == _ts("2023-06-01 10:30")]
    assert later["wind_mw"].tolist() == [1000.0, 1500.0, 2000.0]


def test_interpolation_skips_gaps():
    v = pd.DataFrame(
        {
            "issued_at": [_ts("2023-06-01 08:30")] * 2,
            "target_time": [_ts("2023-06-02 00:00"), _ts("2023-06-02 03:00")],
            "wind_mw": [1.0, 2.0],
        }
    )
    assert len(interpolate_half_hourly(v, "wind_mw")) == 2


@pytest.mark.parametrize(
    ("loader", "handler", "col"),
    [
        (load_demand_forecast, ndf_handler, "demand_mw"),
        (load_wind_forecast, windfor_handler, "wind_mw"),
    ],
)
def test_point_in_time_features_cover_full_delivery_day(settings, loader, handler, col):
    """End to end: load vintages for one BST delivery day, select point-in-time, no leaks."""
    day = dt.date(2023, 6, 2)
    periods = day_periods(day)
    vintages = loader(
        periods[0], periods[-1] + pd.Timedelta(minutes=30), settings, session=FakeSession(handler)
    )

    assert set(vintages.columns) == {"issued_at", "target_time", col}
    # The lookback pulled in D-1 publishes even though the target window starts on D.
    assert vintages["issued_at"].min() < periods[0]

    features = select_vintage(vintages)
    assert features.index.equals(periods.rename("target_time"))  # every half-hour covered
    assert_available_at_issue(features)
    assert (features["issued_at"] <= issue_time(day)).all()


def test_demand_selection_uses_day_ahead_vintage_for_later_periods(settings):
    day = dt.date(2023, 6, 2)
    periods = day_periods(day)
    vintages = load_demand_forecast(
        periods[0],
        periods[-1] + pd.Timedelta(minutes=30),
        settings,
        session=FakeSession(ndf_handler),
    )
    features = select_vintage(vintages)
    # Cutoff is 09:00 UTC on 1 June. Early periods come from the 08:45 rolling vintage; from
    # 04:00 UTC on, only the 07:45 day-ahead vintage reaches that far.
    assert features.loc[_ts("2023-06-02 03:30"), "issued_at"] == _ts("2023-06-01 08:45")
    assert features.loc[_ts("2023-06-02 04:00"), "issued_at"] == _ts("2023-06-01 07:45")


@pytest.mark.network
@pytest.mark.parametrize("loader", [load_demand_forecast, load_wind_forecast])
def test_live_point_in_time_coverage(settings, loader):
    day = dt.date(2023, 6, 2)
    periods = day_periods(day)
    vintages = loader(
        periods[0], periods[-1] + pd.Timedelta(minutes=30), settings, session=make_session()
    )
    features = select_vintage(vintages)
    assert len(features) == 48
    assert_available_at_issue(features)
