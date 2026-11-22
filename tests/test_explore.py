"""Exploration features: issue-time safety, transforms, summaries and catalogue consistency."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gb_price_sandbox.features import explore
from gb_price_sandbox.features.dataset import add_calendar_and_lag_features, london_midnight
from gb_price_sandbox.features.vintage import cutoff_times
from gb_price_sandbox.timing import LONDON, PERIOD

#: Variants deliberately not known at the D-1 issue time (shown as reference only).
NOT_AT_ISSUE = {v.column for v in explore.all_variants() if not v.known_at_issue}
#: Features from external sources; exercised with fake data below.
WEATHER_COLS = {"solar_irradiance_wm2", "temperature_c", "heating_degrees"}


def _dataset(first=dt.date(2024, 3, 1), last=dt.date(2024, 4, 20), seed=0):
    """A small model-dataset lookalike spanning the spring clock change."""
    grid = pd.date_range(
        london_midnight(first), london_midnight(last), freq=PERIOD, inclusive="left"
    )
    rng = np.random.default_rng(seed)
    n = len(grid)
    frame = pd.DataFrame(
        {
            "price": 80 + 30 * rng.standard_normal(n),
            "volume_mwh": 500.0,
            "demand_mw": 30_000 + 3_000 * rng.standard_normal(n),
            "wind_mw": 8_000 + 2_000 * rng.standard_normal(n),
        },
        index=grid,
    )
    frame["residual_demand_mw"] = frame["demand_mw"] - frame["wind_mw"]
    return add_calendar_and_lag_features(frame)


def _extras(index, seed=1):
    rng = np.random.default_rng(seed)
    weather = pd.DataFrame(
        {
            "solar_irradiance_wm2": rng.uniform(0, 600, len(index)),
            "temperature_c": rng.normal(10, 5, len(index)),
        },
        index=index,
    )
    solar = pd.DataFrame({"solar_mw": rng.uniform(0, 9000, len(index))}, index=index)
    return weather, solar


def _build(data, **kw):
    weather, solar = _extras(data.index)
    return explore.add_exploration_features(data, weather=weather, solar=solar, **kw)


def test_every_catalogue_column_is_built():
    out = _build(_dataset())
    missing = [v.column for v in explore.all_variants() if v.column not in out.columns]
    assert not missing


def test_catalogue_consistent():
    keys = [f.key for f in explore.CATALOGUE]
    assert len(keys) == len(set(keys))
    columns = [v.column for v in explore.all_variants()]
    assert len(columns) == len(set(columns)), "a column belongs to two pages"
    for f in explore.CATALOGUE:
        assert f.explanation.strip() and f.timing.strip() and f.summary.strip()
        assert f.shown(), f"{f.key} charts nothing"
        for v in f.variants:
            if v.kind == "categorical":
                assert v.categories, v.column


def test_every_page_cites_its_sources_with_links():
    for f in explore.CATALOGUE:
        assert f.links, f"{f.key} has no source link"
        note = f.source_note()
        assert note.startswith("The raw data was sourced from ")
        assert "for more details." in note
        for d in f.links:
            assert d.url.startswith("https://")
            assert f"[{d.link_text}]({d.url})" in note


def test_lag_profile_covers_allowed_lags_and_flags_used_ones():
    out = _build(_dataset())
    profile = explore.lag_profile(out)
    assert list(profile["lag_days"]) == list(range(2, 15))
    assert set(profile.loc[profile["used"], "lag_days"]) == {2, 3, 7}
    assert profile["correlation"].between(-1, 1).all()


def test_every_lag_column_is_exactly_k_days_earlier():
    data = _dataset()
    out = _build(data)
    for k in explore.LAG_PROFILE_DAYS:
        np.testing.assert_array_equal(
            out[f"price_lag_{k}d"].to_numpy()[48 * k :], data["price"].to_numpy()[: -48 * k]
        )


def test_matrix_short_labels_unique():
    shorts = [v.short for v in explore.matrix_columns(include_transforms=True)]
    assert len(shorts) == len(set(shorts))


@pytest.mark.parametrize("target_day", [dt.date(2024, 3, 31), dt.date(2024, 4, 10)])
def test_no_feature_changes_when_post_cutoff_prices_change(target_day, monkeypatch):
    """Perturbation test: scramble every price that isn't yet published at the issue time.

    For delivery day D, the forecast is issued at 10:00 on D-1. Every price for a period
    ending after 09:00 on D-1 (the publication margin) is replaced, and no feature for day D
    may move. Covers the clock-change day (31 March 2024) too.
    """
    data = _dataset()
    base = _build(data)
    cutoff = cutoff_times(pd.DatetimeIndex([london_midnight(target_day)]))[0]
    published_by = cutoff - pd.Timedelta(hours=explore.PUBLICATION_MARGIN_H)
    late = data.index + PERIOD > published_by

    perturbed = data.copy()
    perturbed.loc[late, "price"] = 1e4 + np.arange(late.sum())
    # Rebuild lags too, exactly as the dataset builder would from the perturbed prices.
    raw = ["price", "volume_mwh", "demand_mw", "wind_mw", "residual_demand_mw"]
    perturbed = add_calendar_and_lag_features(perturbed[raw])
    # The asinh centring uses the whole sample (display only), so pin it: then only genuine
    # leakage can move an asinh feature.
    median, mad = explore.robust_centre(data["price"])
    monkeypatch.setattr(explore, "robust_centre", lambda _p: (median, mad))
    moved = _build(perturbed)

    rows = base["delivery_day"] == target_day
    exempt = {"price", "price_asinh"} | NOT_AT_ISSUE
    cols = [v.column for v in explore.all_variants() if v.column not in exempt]
    pd.testing.assert_frame_equal(base.loc[rows, cols], moved.loc[rows, cols])


def test_d1_morning_uses_only_published_periods():
    data = _dataset()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    # Price = minutes since the epoch, so the "last" value reveals which period it came from.
    data["price"] = ((data.index - epoch) // pd.Timedelta(minutes=1)).astype(float)
    out = _build(data)
    cut = cutoff_times(out.index) - pd.Timedelta(hours=explore.PUBLICATION_MARGIN_H)
    cut_min = pd.Series(((cut - epoch) // pd.Timedelta(minutes=1)).to_numpy(), index=out.index)
    last = out["price_d1_last"].dropna()
    assert len(last) > 0
    assert (last + 30 <= cut_min[last.index]).all()
    # And it is the 08:30 London period of D-1.
    src = pd.to_datetime(last.to_numpy() * 60, unit="s", utc=True).tz_convert(LONDON)
    assert set(zip(src.hour, src.minute, strict=True)) == {(8, 30)}


def test_d2_shape_and_volatility_values():
    data = _dataset()
    out = _build(data)
    day = dt.date(2024, 4, 12)
    d2 = data.loc[data["delivery_day"] == day - dt.timedelta(days=2), "price"]
    row = out[out["delivery_day"] == day].iloc[0]
    assert row["price_d2_max"] == pytest.approx(d2.max())
    assert row["price_d2_spread"] == pytest.approx(d2.max() - d2.min())
    window = data["delivery_day"].between(day - dt.timedelta(days=8), day - dt.timedelta(days=2))
    assert row["price_vol_7d"] == pytest.approx(data.loc[window, "price"].std())
    assert row["price_neg_share_7d"] == pytest.approx((data.loc[window, "price"] < 0).mean())


def test_asinh_handles_negative_prices_and_is_monotone():
    x = np.array([-500.0, -10.0, 0.0, 50.0, 3000.0])
    z = explore.asinh_scale(x, median=50.0, mad=20.0)
    assert np.all(np.diff(z) > 0)
    assert z[3] == 0.0


def test_day_types_prioritise_christmas_then_holidays():
    data = _dataset(dt.date(2023, 12, 20), dt.date(2024, 1, 5))
    out = _build(data, holidays={dt.date(2023, 12, 25), dt.date(2024, 1, 1)})
    by_day = out.groupby("delivery_day")["day_type"].first()
    assert by_day[dt.date(2023, 12, 25)] == 3  # Christmas period beats bank holiday
    assert by_day[dt.date(2023, 12, 23)] == 1  # Saturday
    assert by_day[dt.date(2023, 12, 21)] == 0  # Thursday
    assert by_day[dt.date(2024, 1, 2)] == 0


def test_heating_degrees_one_sided():
    data = _dataset()
    out = _build(data)
    hot = out["temperature_c"] > explore.HEATING_BASE_C
    assert (out.loc[hot, "heating_degrees"] == 0).all()
    cold = out["temperature_c"] < explore.HEATING_BASE_C
    expected = explore.HEATING_BASE_C - out.loc[cold, "temperature_c"]
    np.testing.assert_allclose(out.loc[cold, "heating_degrees"], expected)


def test_missing_sources_give_empty_columns():
    out = explore.add_exploration_features(_dataset())
    assert out[list(WEATHER_COLS)].isna().all().all()
    assert out["solar_outturn_gw"].isna().all()
    assert out["day_type"].isin([0, 1]).all()  # no holidays in March to April without a list


def test_binned_relationship_recovers_monotone_trend():
    x = pd.Series(np.arange(1000, dtype=float))
    y = 2 * x
    table = explore.binned_relationship(x, y, bins=10)
    assert len(table) == 10
    assert table["n"].sum() == 1000
    assert table["p50"].is_monotonic_increasing


def test_histogram_shares_and_clipping():
    # 2% of values are huge spikes; clipping 2.5% per tail keeps them off the plotted range.
    values = pd.Series(np.r_[np.linspace(0, 1, 980), np.full(20, 1e6)])
    table, outside, zeros = explore.histogram(values, bins=5, clip=0.025)
    assert table["right"].max() < 2
    assert table["share"].sum() + outside == pytest.approx(1.0)
    assert outside >= 0.02
    assert zeros == 0.0


def test_histogram_reports_zero_spike_separately():
    values = pd.Series(np.r_[np.zeros(500), np.linspace(1, 10, 500)])
    table, outside, zeros = explore.histogram(values, bins=10, clip=0.0)
    assert zeros == pytest.approx(0.5)
    assert table["left"].min() >= 1
    assert table["share"].sum() + outside + zeros == pytest.approx(1.0)


def test_binned_relationship_merges_tied_bins():
    x = pd.Series(np.r_[np.zeros(500), np.linspace(1, 10, 500)])
    table = explore.binned_relationship(x, x, bins=20)
    assert (table["x"] == 0).sum() == 1


def test_by_category_labels_and_counts():
    codes = pd.Series([0, 0, 1, 1, 1])
    y = pd.Series([1.0, 3.0, 10.0, 20.0, 30.0])
    table = explore.by_category(codes, y, ("a", "b"))
    assert list(table.index) == ["a", "b"]
    assert table.loc["b", "mean"] == pytest.approx(20.0)
    assert table.loc["a", "n"] == 2


def test_spearman_matrix_invariant_to_asinh():
    out = _build(_dataset())
    m = explore.correlation_matrix(out, ["price_lag_2d", "price_lag_2d_asinh", "price"])
    assert m.loc["price_lag_2d", "price"] == pytest.approx(m.loc["price_lag_2d_asinh", "price"])
    assert m.loc["price_lag_2d", "price_lag_2d_asinh"] == pytest.approx(1.0)
