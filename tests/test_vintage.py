"""Leakage tests: features must only use forecasts published by the D-1 issue time."""

import numpy as np
import pandas as pd
import pytest

from gb_price_sandbox.features.vintage import (
    assert_available_at_issue,
    cutoff_times,
    select_vintage,
)

UTC = "UTC"


def _ts(s):
    return pd.Timestamp(s, tz=UTC)


def _frame(rows):
    return pd.DataFrame(rows, columns=["target_time", "issued_at", "wind_mw"])


# Delivery day 2024-01-02 (GMT): cutoff is 2024-01-01 10:00 UTC.
TARGET = _ts("2024-01-02 12:00")


def test_picks_latest_vintage_before_cutoff():
    f = _frame(
        [
            (TARGET, _ts("2023-12-31 09:00"), 100.0),
            (TARGET, _ts("2024-01-01 08:30"), 200.0),
            (TARGET, _ts("2024-01-01 10:00"), 300.0),  # exactly at cutoff: allowed
        ]
    )
    out = select_vintage(f)
    assert out.loc[TARGET, "wind_mw"] == 300.0


def test_ignores_vintages_after_cutoff():
    f = _frame(
        [
            (TARGET, _ts("2024-01-01 08:30"), 200.0),
            (TARGET, _ts("2024-01-01 10:01"), 999.0),  # one minute late: leak
            (TARGET, _ts("2024-01-02 11:00"), 999.0),  # published on the delivery day
        ]
    )
    out = select_vintage(f)
    assert out.loc[TARGET, "wind_mw"] == 200.0
    assert out.loc[TARGET, "issued_at"] == _ts("2024-01-01 08:30")


def test_target_with_only_late_vintages_is_dropped():
    f = _frame([(TARGET, _ts("2024-01-03 08:00"), 999.0)])
    assert select_vintage(f).empty


def test_cutoff_uses_local_delivery_day_across_bst():
    # 2024-06-01 00:30 BST is 2024-05-31 23:30 UTC but belongs to delivery day 1 June, so
    # the cutoff is 31 May 10:00 BST = 09:00 UTC, not 30 May.
    target = _ts("2024-05-31 23:30")
    assert cutoff_times(pd.DatetimeIndex([target]))[0] == _ts("2024-05-31 09:00")
    f = _frame(
        [
            (target, _ts("2024-05-31 08:59"), 1.0),
            (target, _ts("2024-05-31 09:01"), 999.0),
        ]
    )
    assert select_vintage(f).loc[target, "wind_mw"] == 1.0


def test_issue_hour_is_configurable():
    f = _frame(
        [
            (TARGET, _ts("2024-01-01 08:30"), 200.0),
            (TARGET, _ts("2024-01-01 11:30"), 300.0),
        ]
    )
    assert select_vintage(f, hour=12).loc[TARGET, "wind_mw"] == 300.0
    assert select_vintage(f, hour=8).empty


def test_randomised_never_leaks_and_is_latest_eligible():
    rng = np.random.default_rng(42)
    targets = pd.date_range("2024-03-25", "2024-04-05", freq="30min", tz=UTC)  # spans BST
    rows = []
    for t in targets:
        for _ in range(5):
            issued = t - pd.Timedelta(hours=float(rng.uniform(-6, 60)))
            rows.append((t, issued.floor("min"), float(rng.normal())))
    f = _frame(rows)

    out = select_vintage(f)
    assert_available_at_issue(out)  # nothing leaks

    cut = dict(zip(targets, cutoff_times(targets), strict=True))
    for t, group in f.groupby("target_time"):
        eligible = group[group["issued_at"] <= cut[t]]
        if eligible.empty:
            assert t not in out.index
        else:
            assert out.loc[t, "issued_at"] == eligible["issued_at"].max()


def test_assert_available_flags_leaks():
    leaky = pd.DataFrame(
        {"wind_mw": [1.0], "issued_at": [_ts("2024-01-01 12:00")]},
        index=pd.DatetimeIndex([TARGET]),
    )
    with pytest.raises(AssertionError, match="leak"):
        assert_available_at_issue(leaky)


def test_naive_timestamps_rejected():
    f = pd.DataFrame(
        {
            "target_time": [pd.Timestamp("2024-01-02 12:00")],
            "issued_at": [pd.Timestamp("2024-01-01 08:00")],
            "wind_mw": [1.0],
        }
    )
    with pytest.raises(ValueError, match="tz-aware"):
        select_vintage(f)


def test_missing_columns_rejected():
    with pytest.raises(KeyError):
        select_vintage(pd.DataFrame({"target_time": [TARGET]}))
