"""Forecast sandbox: feature groups, model options, and the Train runner."""

import datetime as dt

import numpy as np
import pytest

from gb_price_sandbox import app_data
from gb_price_sandbox.backtest import walk_forward
from gb_price_sandbox.features import explore
from gb_price_sandbox.features.sets import CORE, FEATURE_GROUPS, FeatureSet
from gb_price_sandbox.models import MODEL_SPECS, build_model
from tests.test_explore import _build, _dataset

ALL_GROUPS = FeatureSet(tuple(g.key for g in FEATURE_GROUPS))


@pytest.fixture(scope="module")
def frame():
    """Exploration frame (every feature group present), Oct 2023 to Mar 2024."""
    return _build(_dataset(dt.date(2023, 10, 1), dt.date(2024, 3, 15)))


# --- feature groups --------------------------------------------------------------------------


def test_every_group_column_exists_and_is_known_at_issue(frame):
    variants = {v.column: v for v in explore.all_variants()}
    weather = {"solar_irradiance_wm2", "temperature_c", "heating_degrees"}
    for group in FEATURE_GROUPS:
        assert set(group.price_like) <= set(group.columns)
        for col in group.columns:
            assert col in frame.columns, col
            assert col not in weather, f"{col} is a same-day weather forecast"
            if col in variants:
                assert variants[col].known_at_issue, col


def test_feature_set_columns_and_label():
    fs = FeatureSet(("calendar", "solar"))
    assert fs.columns[: len(CORE.columns)] == CORE.columns
    assert fs.extra == ("is_holiday", "doy_sin", "doy_cos", "solar_lag_2d_gw")
    assert fs.label == "Core + Holidays and season + Solar (2 days earlier)"


# --- model options ---------------------------------------------------------------------------

TEST_START, TEST_END = dt.date(2024, 3, 1), dt.date(2024, 3, 8)


@pytest.mark.parametrize("spec", [s for s in MODEL_SPECS if "objective" in s.options], ids=str)
def test_mean_objective_gives_point_forecasts(frame, spec):
    out = walk_forward(
        frame,
        lambda: build_model(spec.name, CORE, "mean"),
        TEST_START,
        TEST_END,
        features=CORE.columns,
    )
    assert out[["q10", "q50", "q90"]].notna().all().all()
    np.testing.assert_allclose(out["q10"], out["q90"])


@pytest.mark.parametrize("spec", [s for s in MODEL_SPECS if "features" in s.options], ids=str)
def test_extra_feature_groups_reach_the_model(frame, spec):
    model = build_model(spec.name, ALL_GROUPS, "quantile")
    out = walk_forward(frame, lambda: model, TEST_START, TEST_END, features=ALL_GROUPS.columns)
    assert out["q50"].notna().mean() > 0.9
    assert model.features == ALL_GROUPS


def test_lear_design_puts_price_groups_on_asinh_scale():
    model = build_model("LEAR", FeatureSet(("recent_prices", "calendar")))
    assert "price_d1_am_mean" in model.design.price_features
    assert "is_holiday" in model.design.exog_features
    assert "price_neg_share_7d" in model.design.exog_features  # a share, not £/MWh


def test_walk_forward_reports_progress(frame):
    calls = []
    walk_forward(
        frame,
        lambda: build_model("Seasonal naive"),
        TEST_START,
        dt.date(2024, 3, 15),
        progress=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(1, 2), (2, 2)]


# --- the Train runner ------------------------------------------------------------------------


@pytest.fixture
def sandbox_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GBPS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(app_data, "default_test_end", lambda: TEST_END)
    monkeypatch.setattr(app_data, "DEFAULT_TEST_START", TEST_START)


def test_run_sandbox_trains_scores_and_caches(frame, sandbox_env):
    config = app_data.SandboxConfig(
        models=("LEAR", "Seasonal naive", "LightGBM"),
        feature_groups=("calendar",),
        holdout="Since Jan 2023",
        sort_by="MAE",
    )
    progress = []
    result = app_data.run_sandbox(config, frame, lambda f, m: progress.append((f, m)))
    assert list(result.results) == ["Seasonal naive", "LEAR", "LightGBM"]  # registry order
    assert list(result.board.index) == list(result.board.sort_values("MAE").index)
    fractions = [f for f, _ in progress]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any("refit 1 of 1" in m for _, m in progress)

    # Same configuration again: served from the cache, so no refits are reported.
    again = []
    app_data.run_sandbox(config, frame, lambda f, m: again.append(m))
    assert not any("refit" in m for m in again)


def test_sandbox_regime_overrides(sandbox_env):
    default = app_data.SandboxConfig(models=("LightGBM",))
    assert default.regime("LightGBM") == (28, None)
    assert default.regime("QRA") == (28, 730)
    weekly = app_data.SandboxConfig(models=(), refit="Weekly", window="Rolling 1 year")
    assert weekly.regime("QRA") == (7, 365)
    assert app_data.SandboxConfig(models=(), holdout="Last 3 months").test_window() == (
        TEST_START,
        TEST_END,
    )


def test_model_settings_describe_options(sandbox_env):
    cfg = app_data.SandboxConfig(
        models=(), feature_groups=("solar",), objective="Mean (point forecast)"
    )
    lgbm = dict(app_data.model_settings("LightGBM", cfg))
    assert lgbm["Features"] == "Core + Solar (2 days earlier)"
    assert lgbm["Objective"] == "Mean (point forecast)"
    assert lgbm["n_estimators"] == "100"
    naive = dict(app_data.model_settings("Seasonal naive", cfg))
    assert naive["Fitting"].startswith("none")
