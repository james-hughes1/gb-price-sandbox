"""Feature-timing tests: every engineered feature must be known at the D-1 issue time."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gb_price_sandbox.features.dataset import (
    add_calendar_and_lag_features,
    check_lag_availability,
    complete_days,
    london_midnight,
)
from gb_price_sandbox.features.vintage import cutoff_times
from gb_price_sandbox.timing import PERIOD


def _grid(first: dt.date, last: dt.date) -> pd.DatetimeIndex:
    return pd.date_range(
        london_midnight(first), london_midnight(last), freq=PERIOD, inclusive="left"
    )


EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _minutes(idx: pd.DatetimeIndex) -> np.ndarray:
    return ((idx - EPOCH) // pd.Timedelta(minutes=1)).to_numpy()


def _frame(first=dt.date(2024, 3, 1), last=dt.date(2024, 4, 15)):
    grid = _grid(first, last)
    # Price = minutes since the epoch, so every lag value reveals the time it came from.
    price = _minutes(grid).astype(float)
    return pd.DataFrame({"price": price}, index=grid)


def test_one_day_lag_is_rejected():
    # The second half of D-1 hasn't happened by the D-1 morning cutoff.
    with pytest.raises(ValueError, match="1 day"):
        check_lag_availability(_grid(dt.date(2024, 1, 1), dt.date(2024, 1, 10)), (1,))


@pytest.mark.parametrize("hour", [6, 10, 12])
def test_default_lags_are_available_for_all_cutoffs_and_dst(hour):
    # Spans both 2024 clock changes.
    check_lag_availability(_grid(dt.date(2024, 3, 1), dt.date(2024, 11, 1)), (2, 3, 7), hour)


def test_lag_values_come_from_before_cutoff():
    out = add_calendar_and_lag_features(_frame())
    cut_minutes = _minutes(cutoff_times(out.index))
    for k in (2, 3, 7):
        source_start = out[f"price_lag_{k}d"].dropna()
        # Source period must have *ended* (start + 30 min) by the cutoff.
        assert (source_start + 30 <= pd.Series(cut_minutes, out.index)[source_start.index]).all()
        # And it is exactly k x 24 h earlier.
        lagged_from = _minutes(out.index[out[f"price_lag_{k}d"].notna()]) - k * 1440
        np.testing.assert_array_equal(source_start.to_numpy(), lagged_from)


def test_mean_d2_uses_only_day_d_minus_2():
    out = add_calendar_and_lag_features(_frame())
    day = dt.date(2024, 3, 20)
    d2 = out[out["delivery_day"] == day - dt.timedelta(days=2)]["price"].mean()
    assert (out.loc[out["delivery_day"] == day, "price_mean_d2"] == d2).all()
    # The first two days have no D-2 inside the frame.
    assert out.loc[out["delivery_day"] == dt.date(2024, 3, 2), "price_mean_d2"].isna().all()


def test_calendar_columns():
    out = add_calendar_and_lag_features(_frame())
    # 31 March 2024 (spring forward) has 46 periods; slots 2-3 (01:00-01:59 local) don't exist.
    spring = out[out["delivery_day"] == dt.date(2024, 3, 31)]
    assert len(spring) == 46
    assert set(range(48)) - set(spring["period_of_day"]) == {2, 3}
    assert out.loc[out["delivery_day"] == dt.date(2024, 3, 4), "day_of_week"].eq(0).all()


def test_complete_days():
    out = add_calendar_and_lag_features(_frame())
    out.loc[out.index[100], "price"] = np.nan
    days = complete_days(out, ["price"])
    assert out["delivery_day"].iloc[100] not in days
    assert len(days) == out["delivery_day"].nunique() - 1
