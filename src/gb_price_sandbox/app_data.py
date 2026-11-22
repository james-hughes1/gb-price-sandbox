"""Entry points the Streamlit pages call: one function per thing a page shows.

Keeps the pages free of logic. Each function is deterministic given its arguments, so the pages
can wrap them in ``st.cache_data``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import requests

from gb_price_sandbox.backtest import (
    backtest_cache_path,
    cached_walk_forward,
    leaderboard,
)
from gb_price_sandbox.config import get_settings
from gb_price_sandbox.data.calendar import bank_holiday_dates, load_bank_holidays
from gb_price_sandbox.data.open_meteo import load_weather
from gb_price_sandbox.data.pv_live import load_solar_outturn
from gb_price_sandbox.features import explore
from gb_price_sandbox.features.dataset import STUDY_START, build_dataset, london_midnight
from gb_price_sandbox.features.sets import CORE, FEATURE_GROUPS, FeatureSet
from gb_price_sandbox.models import (
    DEFAULT_MODELS,
    FAMILIES,
    MODEL_SPECS,
    MODEL_SPECS_BY_NAME,
    MODELS,
    build_model,
)
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
        if backtest_cache_path(
            data,
            name,
            factory,
            test_start,
            test_end,
            settings,
            retrain_every_days=MODEL_SPECS_BY_NAME[name].retrain_every_days,
            max_train_days=MODEL_SPECS_BY_NAME[name].max_train_days,
        ).exists()
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
        name: cached_walk_forward(
            data,
            name,
            MODELS[name],
            test_start,
            test_end,
            settings,
            retrain_every_days=MODEL_SPECS_BY_NAME[name].retrain_every_days,
            max_train_days=MODEL_SPECS_BY_NAME[name].max_train_days,
        )
        for name in names
    }


def forecast_scores(results: dict[str, pd.DataFrame], sort_by: str = "CRPS") -> pd.DataFrame:
    """Leaderboard for a set of backtest results, best first by ``sort_by``."""
    return leaderboard(results, sort_by)


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


# --- Forecast sandbox ------------------------------------------------------------------------

#: Training-window choices: label -> trailing days (``"default"`` keeps each model's own).
WINDOW_OPTIONS: dict[str, int | None | str] = {
    "Model default": "default",
    "Expanding (all history)": None,
    "Rolling 1 year": 365,
    "Rolling 2 years": 730,
}
#: Holdout choices: label -> months before the end of the test window (None: since 2023).
HOLDOUT_OPTIONS: dict[str, int | None] = {
    "Last 3 months": 3,
    "Last 6 months": 6,
    "Last 12 months": 12,
    "Since Jan 2023": None,
}
#: Refit choices: label -> days (``"default"`` keeps each model's own interval).
REFIT_OPTIONS: dict[str, int | str] = {
    "Model default": "default",
    "Weekly": 7,
}
OBJECTIVE_OPTIONS: dict[str, str] = {
    "Quantiles (median + 80% band)": "quantile",
    "Mean (point forecast)": "mean",
}


@dataclass(frozen=True)
class SandboxConfig:
    """Everything the Train button runs: models, features, regime, objective, scoring."""

    models: tuple[str, ...]
    feature_groups: tuple[str, ...] = ()
    window: str = "Model default"
    holdout: str = "Last 12 months"
    refit: str = "Model default"
    objective: str = "Quantiles (median + 80% band)"
    sort_by: str = "CRPS"

    @property
    def features(self) -> FeatureSet:
        """The feature set handed to the models."""
        return FeatureSet(self.feature_groups)

    @property
    def objective_key(self) -> str:
        """``"quantile"`` or ``"mean"``."""
        return OBJECTIVE_OPTIONS[self.objective]

    def test_window(self) -> tuple[dt.date, dt.date]:
        """Holdout delivery days ``[start, end)``."""
        end = default_test_end()
        months = HOLDOUT_OPTIONS[self.holdout]
        if months is None:
            return DEFAULT_TEST_START, end
        start = (pd.Timestamp(end) - pd.DateOffset(months=months)).date()
        return max(start, DEFAULT_TEST_START), end

    def regime(self, name: str) -> tuple[int, int | None]:
        """``(refit every N days, trailing window days or None)`` for model ``name``."""
        spec = MODEL_SPECS_BY_NAME[name]
        refit = REFIT_OPTIONS[self.refit]
        window = WINDOW_OPTIONS[self.window]
        return (
            spec.retrain_every_days if refit == "default" else int(refit),
            spec.max_train_days if window == "default" else window,
        )


@dataclass(frozen=True)
class SandboxResult:
    """What a Train run produced."""

    config: SandboxConfig
    results: dict[str, pd.DataFrame]
    board: pd.DataFrame
    test_start: dt.date
    test_end: dt.date


def run_sandbox(
    config: SandboxConfig,
    data: pd.DataFrame,
    progress: Callable[[float, str], None] | None = None,
) -> SandboxResult:
    """Backtest every chosen model under ``config`` (cached per model and configuration).

    Args:
        config: The sandbox options.
        data: The exploration frame (:func:`load_feature_data`), which has every feature group.
        progress: Called as ``progress(fraction_done, message)`` as refits complete.
    """
    test_start, test_end = config.test_window()
    settings = get_settings()
    features = config.features
    names = [n for n in MODELS if n in config.models]
    results = {}
    for i, name in enumerate(names):
        retrain, window = config.regime(name)

        def report(done: int, total: int, i: int = i, name: str = name) -> None:
            if progress is not None:
                progress((i + done / total) / len(names), f"{name}: refit {done} of {total}")

        if progress is not None:
            progress(i / len(names), f"{name}: starting")
        uses_features = "features" in MODEL_SPECS_BY_NAME[name].options
        results[name] = cached_walk_forward(
            data,
            name,
            lambda name=name: build_model(name, features, config.objective_key),
            test_start,
            test_end,
            settings,
            retrain_every_days=retrain,
            max_train_days=window,
            features=features.columns if uses_features else CORE.columns,
            progress=report,
        )
    if progress is not None:
        progress(1.0, "Done")
    board = leaderboard(results, config.sort_by) if results else pd.DataFrame()
    return SandboxResult(config, results, board, test_start, test_end)


def feature_group_options() -> list[tuple[str, str, str]]:
    """``(key, label, description)`` of each optional feature group."""
    return [(g.key, g.label, g.description) for g in FEATURE_GROUPS]


def model_options() -> list[tuple[str, str, str, frozenset[str]]]:
    """``(name, family, summary, options it responds to)`` for every model."""
    return [(s.name, s.family, s.summary, s.options) for s in MODEL_SPECS]


def sandbox_families(result: SandboxResult) -> list[str]:
    """Families with at least one trained model, in display order."""
    present = {MODEL_SPECS_BY_NAME[n].family for n in result.results}
    return [f for f in FAMILIES if f in present]


def family_models(result: SandboxResult, family: str) -> list[str]:
    """Trained models in ``family``, in registry order."""
    return [n for n in result.results if MODEL_SPECS_BY_NAME[n].family == family]


def model_settings(name: str, config: SandboxConfig) -> list[tuple[str, str]]:
    """Human-readable settings a model ran with, for the family tab."""
    spec = MODEL_SPECS_BY_NAME[name]
    if spec.family == "Naive baselines":
        return [("Fitting", "none: a fixed rule"), ("Features, objective", "fixed (options n/a)")]
    model = build_model(name, config.features, config.objective_key)
    retrain, window = config.regime(name)
    rows = [
        ("Refit every", f"{retrain} days"),
        ("Training window", "all history" if window is None else f"last {window} days"),
    ]
    if spec.family in ("MSTL", "SARIMAX"):
        rows[1] = ("Training window", f"own {getattr(model, 'window_days', 28)}-day window")
    rows.append(
        (
            "Features",
            config.features.label if "features" in spec.options else "fixed (option n/a)",
        )
    )
    if "objective" in spec.options:
        rows.append(("Objective", config.objective))
    else:
        rows.append(("Objective", "point forecast (option n/a)" if not spec.probabilistic else ""))
    for key, value in sorted(getattr(model, "params", {}).items()):
        rows.append((key, str(value)))
    for attr in ("order", "periods", "daily_harmonics", "weekly_harmonics"):
        if hasattr(model, attr):
            rows.append((attr, str(getattr(model, attr))))
    return [(k, v) for k, v in rows if v]
