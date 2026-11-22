"""New forecast models: registry, naive baselines, quantile models, time-series models."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gb_price_sandbox import app_data
from gb_price_sandbox.backtest import model_fingerprint, walk_forward
from gb_price_sandbox.backtest.walk_forward import published_cutoff
from gb_price_sandbox.features.dataset import FEATURE_COLUMNS
from gb_price_sandbox.models import (
    FAMILIES,
    MODEL_SPECS,
    MODELS,
    QUANTILE_COLUMNS,
    D2Persistence,
    WeekdayProfile,
    sort_quantiles,
)
from gb_price_sandbox.models import linear_quantile as lq
from gb_price_sandbox.timing import PERIOD
from tests.test_backtest import synthetic_dataset

TEST_START, TEST_END = dt.date(2024, 3, 1), dt.date(2024, 3, 8)


@pytest.fixture(scope="module")
def data():
    # Long enough for QRA's 90-day minimum plus its calibration window.
    return synthetic_dataset(dt.date(2023, 10, 1), dt.date(2024, 3, 8))


_BACKTESTS: dict[str, pd.DataFrame] = {}


def backtest(data, name):
    """Walk-forward result for ``name`` on the module dataset, computed once per session."""
    if name not in _BACKTESTS:
        _BACKTESTS[name] = walk_forward(data, MODELS[name], TEST_START, TEST_END)
    return _BACKTESTS[name]


# --- registry --------------------------------------------------------------------------------


def test_registry_names_unique_and_match_models():
    names = [s.name for s in MODEL_SPECS]
    assert len(names) == len(set(names))
    for spec in MODEL_SPECS:
        assert spec.factory().name == spec.name


def test_families_contiguous_and_fit_the_palette():
    order = [s.family for s in MODEL_SPECS]
    for fam in FAMILIES:
        idx = [i for i, f in enumerate(order) if f == fam]
        assert idx == list(range(idx[0], idx[-1] + 1)), f"{fam} is split"
    assert len(FAMILIES) <= 8  # one palette slot each, never cycled


def test_colours_follow_family_and_dashes_separate_siblings():
    colors, dashes = app_data.model_colors(dark=False), app_data.model_dashes()
    by_family: dict[str, list[str]] = {}
    for s in MODEL_SPECS:
        by_family.setdefault(s.family, []).append(s.name)
    family_colours = [colors[names[0]] for names in by_family.values()]
    assert len(set(family_colours)) == len(family_colours)
    for names in by_family.values():
        assert len({colors[n] for n in names}) == 1
        assert len({dashes[n] for n in names}) == len(names)


def test_fingerprint_separates_presets_and_is_stable():
    light, heavy = MODELS["LightGBM (light)"], MODELS["LightGBM (heavy)"]
    assert model_fingerprint(light) != model_fingerprint(heavy)
    assert model_fingerprint(light) == model_fingerprint(light)
    assert model_fingerprint(MODELS["LEAR"]) == model_fingerprint(MODELS["LEAR"])


# --- naive baselines -------------------------------------------------------------------------


def test_d2_persistence_is_the_two_day_lag(data):
    pred = D2Persistence().predict_quantiles(data)
    pd.testing.assert_series_equal(pred["q50"], data["price_lag_2d"], check_names=False)


def test_weekday_profile_averages_available_weekday_lags():
    X = pd.DataFrame(
        {
            "price_lag_7d": [10.0, 10.0, np.nan],
            "price_lag_14d": [20.0, np.nan, np.nan],
            "price_lag_21d": [30.0, 30.0, np.nan],
            "price_lag_28d": [40.0, 50.0, np.nan],
        }
    )
    q50 = WeekdayProfile().predict_quantiles(X)["q50"]
    assert q50.iloc[0] == 25.0
    assert q50.iloc[1] == 30.0  # mean of the three present
    assert np.isnan(q50.iloc[2])


def test_sort_quantiles_uncrosses_rows_and_keeps_missing():
    frame = pd.DataFrame({"q10": [5.0, np.nan], "q50": [3.0, 1.0], "q90": [4.0, 2.0]})
    out = sort_quantiles(frame)
    assert list(out.iloc[0]) == [3.0, 4.0, 5.0]
    assert np.isnan(out.iloc[1]["q10"])


# --- every model through the walk-forward ----------------------------------------------------


@pytest.mark.parametrize("name", [s.name for s in MODEL_SPECS])
def test_every_model_backtests_with_ordered_quantiles(data, name):
    out = backtest(data, name)
    q = out[list(QUANTILE_COLUMNS)]
    assert len(out) == 7 * 48
    assert q.notna().all().all(), f"{name} left gaps"
    assert (q["q10"] <= q["q50"]).all() and (q["q50"] <= q["q90"]).all()


@pytest.mark.parametrize("name", [s.name for s in MODEL_SPECS if s.probabilistic])
def test_probabilistic_models_give_real_bands(data, name):
    out = backtest(data, name)
    width = (out["q90"] - out["q10"]).mean()
    coverage = ((out["y"] >= out["q10"]) & (out["y"] <= out["q90"])).mean()
    assert width > 0
    assert 0.4 <= coverage <= 1.0, f"{name} coverage {coverage:.2f}"


def test_learning_models_beat_seasonal_naive(data):
    naive = backtest(data, "Seasonal naive")
    base = (naive["q50"] - naive["y"]).abs().mean()
    for name in ("LEAR", "Linear quantile regression", "QRA", "LightGBM (light)"):
        out = backtest(data, name)
        assert (out["q50"] - out["y"]).abs().mean() < base, name


# --- leakage: history models and QRA calibration ---------------------------------------------


class HistorySpy:
    """A history model that records what the backtest hands it."""

    name = "History spy"
    uses_history = True
    calls: list = []

    def fit(self, X, y):
        return self

    def predict_day(self, history, future):
        HistorySpy.calls.append((history.copy(), future.copy()))
        return pd.DataFrame(0.0, index=future.index, columns=list(QUANTILE_COLUMNS))


def test_history_models_only_see_published_prices(data):
    HistorySpy.calls = []
    walk_forward(data, HistorySpy, TEST_START, TEST_END)
    assert len(HistorySpy.calls) == 7
    for day, (history, future) in zip(
        pd.date_range(TEST_START, periods=7).date, HistorySpy.calls, strict=True
    ):
        cutoff = published_cutoff(day)
        assert history.index.max() + PERIOD <= cutoff  # last price ended by 09:00 on D-1
        assert history.index.max() + PERIOD > cutoff - PERIOD  # ...and nothing older is skipped
        assert future.index[0] == history.index.max() + PERIOD  # contiguous
        assert "price" not in future.columns
        assert set(future.columns) == set(FEATURE_COLUMNS)
        assert (data.loc[future.index[-1], "delivery_day"]) == day


def test_qra_calibration_forecasts_are_out_of_sample(data, monkeypatch):
    seen = []

    class RecordingLEAR(lq.LEAR):
        def fit(self, X, y):
            seen.append(pd.Index(lq.delivery_day(X.index)).max())
            return super().fit(X, y)

    monkeypatch.setattr(lq, "LEAR", RecordingLEAR)
    train = data[data["delivery_day"] <= dt.date(2024, 2, 28)]
    model = lq.QRA().fit(train[list(FEATURE_COLUMNS)], train["price"])
    cal_start = model.calibration_start_
    # One LEAR per calibration block, each trained only up to two days before its block,
    # then a final one on everything.
    blocks = [
        cal_start + dt.timedelta(days=k)
        for k in range(0, (dt.date(2024, 2, 28) - cal_start).days + 1, lq.QRA_REFIT_DAYS)
    ]
    assert seen[:-1] == [b - dt.timedelta(days=2) for b in blocks]
    assert seen[-1] == dt.date(2024, 2, 28)
