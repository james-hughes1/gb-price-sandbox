import datetime as dt
import importlib

import numpy as np
import pandas as pd
import pytest

from gb_price_sandbox.backtest import (
    cached_walk_forward,
    common_rows,
    error_by_period_of_day,
    leaderboard,
    model_fingerprint,
    training_mask,
    walk_forward,
)
from gb_price_sandbox.config import Settings
from gb_price_sandbox.features.dataset import (
    FEATURE_COLUMNS,
    add_calendar_and_lag_features,
    london_midnight,
)
from gb_price_sandbox.models import LEAR, QUANTILE_COLUMNS, SeasonalNaive, point_as_quantiles
from gb_price_sandbox.optimise import PERFECT_FORESIGHT, BatterySpec, battery_backtest
from gb_price_sandbox.timing import PERIOD, issue_time


def synthetic_dataset(first=dt.date(2024, 1, 1), last=dt.date(2024, 4, 1), seed=0):
    """Prices driven by residual demand plus a daily shape and noise."""
    rng = np.random.default_rng(seed)
    grid = pd.date_range(
        london_midnight(first), london_midnight(last), freq=PERIOD, inclusive="left"
    )
    n = len(grid)
    hour = grid.tz_convert("Europe/London").hour.to_numpy()
    demand = 25_000 + 5_000 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 500, n)
    wind = np.clip(8_000 + np.cumsum(rng.normal(0, 300, n)), 500, 20_000)
    price = 0.006 * (demand - wind) + rng.normal(0, 5, n)
    frame = pd.DataFrame(
        {"price": price, "volume_mwh": 500.0, "demand_mw": demand, "wind_mw": wind}, index=grid
    )
    frame["residual_demand_mw"] = frame["demand_mw"] - frame["wind_mw"]
    return add_calendar_and_lag_features(frame)


class Spy:
    """Records what the backtest hands the model."""

    name = "Spy"
    seen: list = []

    def fit(self, X, y):
        Spy.seen.append({"columns": list(X.columns), "last_train": X.index.max()})
        return self

    def predict_quantiles(self, X):
        return point_as_quantiles(pd.Series(0.0, index=X.index))


# --- models ----------------------------------------------------------------------------------


def test_seasonal_naive_returns_week_lag():
    data = synthetic_dataset()
    pred = SeasonalNaive().fit(data, data["price"]).predict_quantiles(data)
    pd.testing.assert_series_equal(pred["q50"], data["price_lag_7d"], check_names=False)
    assert (pred["q10"].dropna() == pred["q90"].dropna()).all()


def test_lear_beats_naive_on_learnable_data():
    data = synthetic_dataset()
    train = data[data["delivery_day"] < dt.date(2024, 3, 1)]
    test = data[data["delivery_day"] >= dt.date(2024, 3, 1)]
    lear = LEAR().fit(train, train["price"]).predict_quantiles(test)["q50"]
    naive = SeasonalNaive().predict_quantiles(test)["q50"]
    err = lambda p: (p - test["price"]).abs().mean()  # noqa: E731
    assert err(lear) < 0.5 * err(naive)


def test_lear_handles_negative_prices_and_missing_inputs():
    data = synthetic_dataset()
    data["price"] -= 120  # mostly negative prices
    data = add_calendar_and_lag_features(
        data[["price", "volume_mwh", "demand_mw", "wind_mw", "residual_demand_mw"]]
    )
    data.iloc[2000, data.columns.get_loc("demand_mw")] = np.nan
    model = LEAR().fit(data, data["price"])
    pred = model.predict_quantiles(data)["q50"]
    assert np.isnan(pred.iloc[2000])
    assert pred.iloc[3000:].notna().all()
    assert pred.iloc[3000:].mean() < 0


# --- walk-forward ----------------------------------------------------------------------------


def test_training_mask_trailing_window():
    data = synthetic_dataset()
    mask = training_mask(data, dt.date(2024, 3, 1), max_train_days=10)
    days = data.loc[mask, "delivery_day"]
    assert days.min() == dt.date(2024, 2, 19) and days.max() == dt.date(2024, 2, 28)


def test_training_mask_stops_two_days_before():
    data = synthetic_dataset()
    mask = training_mask(data, dt.date(2024, 2, 10))
    assert data.loc[mask, "delivery_day"].max() == dt.date(2024, 2, 8)


def test_walk_forward_never_trains_on_the_future_and_hides_realised_columns():
    data = synthetic_dataset()
    Spy.seen = []
    out = walk_forward(data, Spy, dt.date(2024, 2, 1), dt.date(2024, 3, 1), retrain_every_days=7)

    assert len(Spy.seen) == 5  # 29 days in blocks of 7
    for call in Spy.seen:
        assert set(call["columns"]) == set(FEATURE_COLUMNS)
        assert "price" not in call["columns"] and "volume_mwh" not in call["columns"]
    # Each block's training data ends by the issue time of the block's first day.
    for call, block_start in zip(
        Spy.seen, pd.date_range("2024-02-01", periods=5, freq="7D").date, strict=True
    ):
        assert call["last_train"] + PERIOD <= issue_time(block_start)
    assert out["delivery_day"].min() == dt.date(2024, 2, 1)
    assert out["delivery_day"].max() == dt.date(2024, 2, 29)
    assert list(out.columns[:3]) == list(QUANTILE_COLUMNS)


def test_walk_forward_detects_leaky_training_window(monkeypatch):
    wf = importlib.import_module("gb_price_sandbox.backtest.walk_forward")

    data = synthetic_dataset()
    monkeypatch.setattr(wf, "training_mask", lambda d, day, *_: d["delivery_day"] < day)
    with pytest.raises(AssertionError, match="issue time"):
        wf.walk_forward(data, Spy, dt.date(2024, 2, 1), dt.date(2024, 2, 8))


def test_cached_walk_forward_reuses_results(tmp_path):
    data = synthetic_dataset()
    settings = Settings(data_dir=tmp_path)
    args = (data, "Seasonal naive", SeasonalNaive, dt.date(2024, 2, 1), dt.date(2024, 2, 15))
    first = cached_walk_forward(*args, settings)
    files = list((tmp_path / "processed" / "backtests").iterdir())
    assert len(files) == 1 and model_fingerprint(SeasonalNaive) in files[0].name
    second = cached_walk_forward(*args, settings)
    pd.testing.assert_frame_equal(first, second)


# --- evaluation ------------------------------------------------------------------------------


def _results(n=96):
    idx = pd.date_range("2024-06-01", periods=n, freq="30min", tz="UTC")
    y = pd.Series(np.linspace(0, 100, n), idx)
    good = point_as_quantiles(y + 1.0).assign(y=y)
    bad = point_as_quantiles(y + 10.0).assign(y=y)
    return {"Good": good, "Bad": bad}


def test_leaderboard_ranks_and_scores():
    board = leaderboard(_results())
    assert list(board.index) == ["Good", "Bad"]
    assert board.loc["Good", "MAE"] == pytest.approx(1.0)
    # Degenerate quantiles: CRPS == MAE, pinball == MAE / 2.
    assert board.loc["Bad", "CRPS"] == pytest.approx(10.0)
    assert board.loc["Bad", "Pinball"] == pytest.approx(5.0)


def test_leaderboard_uses_common_rows():
    res = _results()
    res["Bad"].iloc[:10, 0:3] = np.nan
    assert len(common_rows(res)) == 86
    assert (leaderboard(res)["Periods"] == 86).all()


def test_error_by_period_of_day():
    res = _results()
    slot = pd.Series(np.arange(96) % 48, index=res["Good"].index)
    table = error_by_period_of_day(res, slot)
    assert table.shape == (48, 2)
    assert np.allclose(table["Bad"], 10.0)


def test_battery_backtest_perfect_is_upper_bound():
    data = synthetic_dataset(first=dt.date(2024, 1, 1), last=dt.date(2024, 2, 1))
    res = {
        "Naive": walk_forward(data, SeasonalNaive, dt.date(2024, 1, 15), dt.date(2024, 1, 29)),
    }
    table, schedules = battery_backtest(res, BatterySpec())
    assert table.loc[PERFECT_FORESIGHT, "Capture rate"] == pytest.approx(1.0)
    assert table.loc["Naive", "Revenue (GBP)"] <= table.loc[PERFECT_FORESIGHT, "Revenue (GBP)"]
    assert set(schedules) == {PERFECT_FORESIGHT, "Naive"}
    assert table["Days"].iloc[0] == 14
