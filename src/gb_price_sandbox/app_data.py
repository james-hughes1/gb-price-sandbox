"""Entry points the Streamlit pages call: one function per thing a page shows.

Keeps the pages free of logic. Each function is deterministic given its arguments, so the pages
can wrap them in ``st.cache_data``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd
import requests

from gb_price_sandbox.backtest import (
    backtest_cache_path,
    cached_walk_forward,
    error_by_period_of_day,
    leaderboard,
)
from gb_price_sandbox.config import get_settings
from gb_price_sandbox.data.calendar import bank_holiday_dates, load_bank_holidays
from gb_price_sandbox.data.open_meteo import load_weather
from gb_price_sandbox.data.pv_live import load_solar_outturn
from gb_price_sandbox.features import explore
from gb_price_sandbox.features.dataset import STUDY_START, build_dataset, london_midnight
from gb_price_sandbox.models import DEFAULT_MODELS, FAMILIES, MODEL_SPECS, MODELS
from gb_price_sandbox.optimise import PERFECT_FORESIGHT, BatterySpec, battery_backtest
from gb_price_sandbox.timing import LONDON
from gb_price_sandbox.viz.theme import ACTUAL, entity_colors

#: First delivery day forecast in backtests. 2022 is the initial training year.
DEFAULT_TEST_START = dt.date(2023, 1, 1)


def today_london() -> dt.date:
    """Current GB calendar date."""
    return pd.Timestamp.now(tz=LONDON).date()


def data_end() -> dt.date:
    """Exclusive end of the dataset: today (so yesterday is the last full day)."""
    return today_london()


def default_test_end() -> dt.date:
    """Exclusive end of the default backtest window: the first of the current month.

    Stops at a month boundary so cached backtests stay valid for the whole month.
    """
    return today_london().replace(day=1)


def load_study_data() -> pd.DataFrame:
    """The model dataset from the study start to yesterday (cached on disk per day)."""
    return build_dataset(STUDY_START, data_end(), get_settings())


def model_colors(dark: bool) -> dict[str, str]:
    """Fixed colour for every model, the realised price, and perfect foresight.

    Colour follows the model *family* (8 families, 8 palette slots, never cycled); models in
    the same family are told apart by line dash (see :func:`model_dashes`) and by name.
    """
    family_colors = entity_colors(list(FAMILIES), dark)
    colors = entity_colors(
        [ACTUAL, PERFECT_FORESIGHT], dark, actual_names=(ACTUAL, PERFECT_FORESIGHT)
    )
    colors.update({s.name: family_colors[s.family] for s in MODEL_SPECS})
    return colors


_DASHES = ("solid", "dash", "dot")


def model_dashes() -> dict[str, str]:
    """Line dash per model: solid for a family's first model, then dashed, then dotted."""
    seen: dict[str, int] = {}
    out = {ACTUAL: "solid", PERFECT_FORESIGHT: "solid"}
    for s in MODEL_SPECS:
        k = seen.get(s.family, 0)
        out[s.name] = _DASHES[k % len(_DASHES)]
        seen[s.family] = k + 1
    return out


def cached_model_names(test_start: dt.date, test_end: dt.date) -> list[str]:
    """Models whose backtest for this window is already on disk (so loads instantly)."""
    data = load_study_data()
    settings = get_settings()
    return [
        name
        for name, factory in MODELS.items()
        if backtest_cache_path(data, name, factory, test_start, test_end, settings).exists()
    ]


def default_model_selection(test_start: dt.date, test_end: dt.date) -> list[str]:
    """The quick models plus any already backtested for this window, in registry order."""
    chosen = set(DEFAULT_MODELS) | set(cached_model_names(test_start, test_end))
    return [name for name in MODELS if name in chosen]


def run_backtests(
    test_start: dt.date, test_end: dt.date, model_names: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Walk-forward results per model (cached on disk)."""
    data = load_study_data()
    names = model_names or list(MODELS)
    settings = get_settings()
    return {
        name: cached_walk_forward(data, name, MODELS[name], test_start, test_end, settings)
        for name in names
    }


def forecast_scores(
    results: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leaderboard and MAE by time of day for a set of backtest results."""
    data = load_study_data()
    return leaderboard(results), error_by_period_of_day(results, data["period_of_day"])


def battery_results(
    results: dict[str, pd.DataFrame],
    battery: BatterySpec,
    window_start: dt.date,
    window_end: dt.date,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Capture-rate table and schedules for delivery days ``[window_start, window_end)``."""
    sliced = {
        name: res[(res["delivery_day"] >= window_start) & (res["delivery_day"] < window_end)]
        for name, res in results.items()
    }
    return battery_backtest(sliced, battery)


#: Battery-page period choices: label -> months back from the end of the test window
#: (``None`` = the whole test window).
BATTERY_WINDOWS: dict[str, int | None] = {
    "Last 3 months": 3,
    "Last 12 months": 12,
    "Whole test period": None,
}


def battery_window(label: str) -> tuple[dt.date, dt.date]:
    """Delivery-day window ``[start, end)`` for a :data:`BATTERY_WINDOWS` label."""
    end = default_test_end()
    months = BATTERY_WINDOWS[label]
    if months is None:
        return DEFAULT_TEST_START, end
    start = (pd.Timestamp(end) - pd.DateOffset(months=months)).date()
    return max(start, DEFAULT_TEST_START), end


def run_battery_page(
    power_mw: float, duration_h: float, rte: float, cycles: float, window_label: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.Series], list[dt.date]]:
    """Everything the Battery page shows for one set of slider values.

    Returns:
        The capture-rate table, per-period schedules, each model's median forecast aligned to
        the schedules, and the list of delivery days available for the day-detail chart.
    """
    battery = BatterySpec(
        power_mw=power_mw,
        energy_mwh=power_mw * duration_h,
        round_trip_efficiency=rte,
        max_cycles_per_day=cycles,
    )
    start, end = battery_window(window_label)
    test_end = default_test_end()
    # Only models that are quick or already backtested, so the page never stalls for minutes.
    names = default_model_selection(DEFAULT_TEST_START, test_end)
    results = run_backtests(DEFAULT_TEST_START, test_end, names)
    table, schedules = battery_results(results, battery, start, end)
    index = schedules[PERFECT_FORESIGHT].index
    forecasts = {name: res.loc[index, "q50"] for name, res in results.items()}
    days = sorted(set(index.tz_convert(LONDON).date))
    return table, schedules, forecasts, days


# --- Features tab ----------------------------------------------------------------------------


def load_feature_data() -> tuple[pd.DataFrame, list[str]]:
    """The study dataset plus every exploration feature (see :mod:`~.features.explore`).

    Weather, solar and holiday sources are optional: if one can't be reached its columns are
    left empty rather than breaking the page.

    Returns:
        ``(frame, unavailable)``: the exploration frame and the names of sources that failed.
    """
    data = load_study_data()
    settings = get_settings()
    start, end = london_midnight(STUDY_START), london_midnight(data_end())
    unavailable: list[str] = []

    def attempt(name, load):
        try:
            return load()
        except (requests.RequestException, KeyError, ValueError):
            unavailable.append(name)
            return None

    weather = attempt("Open-Meteo weather", lambda: load_weather(start, end, settings))
    solar = attempt("PV_Live solar", lambda: load_solar_outturn(start, end, settings))
    holidays = attempt(
        "GOV.UK bank holidays",
        lambda: bank_holiday_dates(load_bank_holidays(settings, until=data_end())),
    )
    frame = explore.add_exploration_features(data, weather=weather, solar=solar, holidays=holidays)
    return frame, unavailable


def feature_selector() -> list[tuple[str, str]]:
    """``(key, label)`` for each feature, in selector order. Labels name the group."""
    group = {"target": "Target", "used": "Used today", "built": "Built", "candidate": "Candidate"}
    return [(f.key, f"{group[f.status]} · {f.title}") for f in explore.ordered_catalogue()]


def feature_filter_bounds(frame: pd.DataFrame) -> tuple[int, int]:
    """First and last calendar year in the exploration frame."""
    years = frame.index.tz_convert(LONDON).year
    return int(years.min()), int(years.max())


def filter_feature_frame(
    frame: pd.DataFrame, years: tuple[int, int], periods: tuple[int, int]
) -> pd.DataFrame:
    """Rows in the year range and local half-hour range (both inclusive).

    Args:
        frame: Exploration frame.
        years: First and last calendar year (local time).
        periods: First and last half-hour of the day, 0 (00:00) to 47 (23:30).
    """
    year = frame.index.tz_convert(LONDON).year
    keep = (year >= years[0]) & (year <= years[1])
    keep &= frame["period_of_day"].between(periods[0], periods[1]).to_numpy()
    return frame[keep]


@dataclass(frozen=True)
class FeatureView:
    """Everything the Features tab draws for one selected feature."""

    spec: explore.FeatureSpec
    variants: list[explore.Variant]
    daily: pd.DataFrame
    histograms: dict[str, tuple[pd.DataFrame, float, float]]
    relationships: dict[str, pd.DataFrame]
    categories: dict[str, pd.DataFrame]
    correlations: dict[str, tuple[float, float]]
    coverage: dict[str, tuple[dt.date, dt.date]]
    lag_profile: pd.DataFrame | None = None


def feature_view(frame: pd.DataFrame, key: str) -> FeatureView:
    """Summaries for feature ``key``: daily series, histograms, price relationships.

    Only the variants charted on the page are summarised; those with no data at all (e.g. a
    source that failed to load) are dropped.
    """
    spec = explore.CATALOGUE_BY_KEY[key]
    variants = [v for v in spec.shown() if frame[v.column].notna().any()]
    continuous = [v for v in variants if v.kind == "continuous"]
    daily = (
        explore.daily_means(frame, [v.column for v in continuous])
        if continuous and spec.timeseries
        else None
    )
    histograms, relationships, categories, correlations, coverage = {}, {}, {}, {}, {}
    for v in variants:
        col = frame[v.column]
        present = frame.loc[col.notna(), "delivery_day"]
        coverage[v.column] = (present.min(), present.max())
        # Codes of a many-valued category (day names, months) have no meaningful order.
        ordered = v.kind == "continuous" or len(v.categories) == 2
        # A column that is constant in the filtered rows has no defined correlation.
        if v.column != v.target and ordered and col.min() < col.max():
            correlations[v.column] = (
                float(col.corr(frame[v.target], method="spearman")),
                float(col.corr(frame[v.target], method="pearson")),
            )
        if v.kind == "continuous":
            histograms[v.column] = explore.histogram(col)
            if v.column != v.target:
                relationships[v.column] = explore.binned_relationship(col, frame[v.target])
        else:
            categories[v.column] = explore.by_category(col, frame[v.target], v.categories)
    return FeatureView(
        spec=spec,
        variants=variants,
        daily=daily if daily is not None else pd.DataFrame(),
        histograms=histograms,
        relationships=relationships,
        categories=categories,
        correlations=correlations,
        coverage=coverage,
        lag_profile=explore.lag_profile(frame) if spec.lag_profile else None,
    )


def feature_correlations(
    frame: pd.DataFrame, method: str, include_transforms: bool
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Correlation matrix for the Features tab.

    Returns:
        ``(matrix, labels, label_to_feature)``: the matrix, its short labels in order, and
        the feature key each label belongs to (for highlighting the selected feature).
    """
    owner = {v.column: f.key for f in explore.CATALOGUE for v in f.variants}
    variants = explore.matrix_columns(include_transforms)
    matrix = explore.correlation_matrix(frame, [v.column for v in variants], method)
    short = {v.column: v.short for v in variants}
    labels = [short[c] for c in matrix.columns]
    return matrix, labels, {short[c]: owner[c] for c in matrix.columns}
