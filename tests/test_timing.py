import datetime as dt

import pandas as pd
import pytest

from gb_price_sandbox.timing import day_periods, delivery_day, ensure_utc_index, issue_time


@pytest.mark.parametrize(
    ("day", "n"),
    [(dt.date(2024, 6, 1), 48), (dt.date(2024, 3, 31), 46), (dt.date(2024, 10, 27), 50)],
)
def test_day_periods_handles_clock_changes(day, n):
    idx = day_periods(day)
    assert len(idx) == n
    assert (delivery_day(idx) == day).all()


def test_summer_day_starts_at_2300_utc():
    assert day_periods(dt.date(2024, 6, 1))[0] == pd.Timestamp("2024-05-31 23:00", tz="UTC")


def test_issue_time_is_d_minus_1_local():
    # BST: 10:00 London on 31 May == 09:00 UTC.
    assert issue_time(dt.date(2024, 6, 1)) == pd.Timestamp("2024-05-31 09:00", tz="UTC")
    # GMT: 10:00 London == 10:00 UTC.
    assert issue_time(dt.date(2024, 1, 2)) == pd.Timestamp("2024-01-01 10:00", tz="UTC")


def test_naive_index_rejected():
    with pytest.raises(ValueError):
        ensure_utc_index(pd.DatetimeIndex(["2024-01-01"]))
