"""Entry points the Streamlit pages call: one function per thing a page shows.

Keeps the pages free of logic. Each function is deterministic given its arguments, so the pages
can wrap them in ``st.cache_data``.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from gb_price_sandbox.backtest import cached_walk_forward, error_by_period_of_day, leaderboard
from gb_price_sandbox.config import get_settings
from gb_price_sandbox.features.dataset import STUDY_START, build_dataset
from gb_price_sandbox.models import MODELS
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
    """Fixed colour for every model, the realised price, and perfect foresight."""
    names = [ACTUAL, PERFECT_FORESIGHT, *MODELS]
    return entity_colors(names, dark, actual_names=(ACTUAL, PERFECT_FORESIGHT))


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
    results = run_backtests(DEFAULT_TEST_START, default_test_end())
    table, schedules = battery_results(results, battery, start, end)
    index = schedules[PERFECT_FORESIGHT].index
    forecasts = {name: res.loc[index, "q50"] for name, res in results.items()}
    days = sorted(set(index.tz_convert(LONDON).date))
    return table, schedules, forecasts, days
