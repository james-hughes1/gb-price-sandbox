"""Walk-forward backtest: expanding window, issued D-1 for delivery day D, weekly retrain."""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import sys
from collections.abc import Callable

import pandas as pd

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.cache import cached_parquet
from gb_price_sandbox.features.dataset import FEATURE_COLUMNS, TARGET
from gb_price_sandbox.models.base import QUANTILE_COLUMNS, Forecaster, HistoryForecaster
from gb_price_sandbox.timing import MID_PUBLICATION_MARGIN, PERIOD, issue_time

#: Days of published history handed to a :class:`HistoryForecaster` for each delivery day.
HISTORY_DAYS = 35


def training_mask(
    data: pd.DataFrame, first_forecast_day: dt.date, max_train_days: int | None = None
) -> pd.Series:
    """Rows usable for a model issued at the cutoff of ``first_forecast_day`` (D-1 morning).

    Only delivery days up to D-2 have fully settled prices by then, so those are the
    training targets. Rows with a missing target are dropped. ``max_train_days`` turns the
    expanding window into a trailing one of that many days.
    """
    last_train_day = first_forecast_day - dt.timedelta(days=2)
    mask = (data["delivery_day"] <= last_train_day) & data[TARGET].notna()
    if max_train_days is not None:
        mask &= data["delivery_day"] > last_train_day - dt.timedelta(days=max_train_days)
    return mask


def published_cutoff(day: dt.date, hour: int = 10) -> pd.Timestamp:
    """Latest period *end* whose price is known at delivery day ``day``'s issue time."""
    return issue_time(day, hour) - MID_PUBLICATION_MARGIN


def _predict_with_history(
    model: HistoryForecaster,
    data: pd.DataFrame,
    block: list[dt.date],
    features: list[str],
    hour: int,
) -> pd.DataFrame:
    """Forecast each day of ``block`` from the prices published by that day's issue time."""
    index = data.index
    span = int(HISTORY_DAYS * 48)
    parts = []
    for day in block:
        published_by = published_cutoff(day, hour)
        # First period whose price is NOT yet published (period end > published_by).
        first_unpublished = index.searchsorted(published_by - PERIOD, side="right")
        day_rows = data.index[data["delivery_day"] == day]
        last = index.searchsorted(day_rows.max(), side="right")
        history = data.iloc[max(0, first_unpublished - span) : first_unpublished]
        history = history[[*features, TARGET]]
        future = data.iloc[first_unpublished:last][features]
        # Leakage guard: no price in the history ends after it could have been published.
        if len(history) and history.index.max() + PERIOD > published_by:
            raise AssertionError(f"history for {day} includes unpublished prices")
        pred = model.predict_day(history, future)
        parts.append(pred.loc[day_rows, list(QUANTILE_COLUMNS)])
    return pd.concat(parts)


def walk_forward(
    data: pd.DataFrame,
    make_model: Callable[[], Forecaster],
    test_start: dt.date,
    test_end: dt.date,
    *,
    retrain_every_days: int = 7,
    max_train_days: int | None = None,
    hour: int = 10,
) -> pd.DataFrame:
    """Backtest a model over delivery days ``[test_start, test_end)``.

    Every ``retrain_every_days`` days a fresh model is fitted on all rows allowed by
    :func:`training_mask` (an expanding window). It then forecasts the next block of days.
    The model sees only :data:`FEATURE_COLUMNS`. A :class:`HistoryForecaster` additionally
    gets, for each delivery day, the prices published by that day's issue time (see
    :func:`published_cutoff`).

    Args:
        data: Output of :func:`gb_price_sandbox.features.dataset.build_dataset`.
        make_model: Zero-argument factory returning an unfitted model.
        test_start: First delivery day forecast (inclusive).
        test_end: Last delivery day forecast (exclusive).
        retrain_every_days: Retrain cadence in days.
        max_train_days: Train on only the last this-many days (default: all history).
        hour: Forecast issue hour, London time on D-1, used for the leakage assertion.

    Returns:
        Frame indexed like ``data`` (test period only) with ``QUANTILE_COLUMNS``, the realised
        ``y``, the ``delivery_day`` and the ``trained_through`` day used for each forecast.
    """
    features = list(FEATURE_COLUMNS)
    in_test = (data["delivery_day"] >= test_start) & (data["delivery_day"] < test_end)
    test_days = sorted(data.loc[in_test, "delivery_day"].unique())
    blocks = [
        test_days[i : i + retrain_every_days] for i in range(0, len(test_days), retrain_every_days)
    ]

    outputs = []
    for block in blocks:
        first = block[0]
        train = data[training_mask(data, first, max_train_days)]
        if train.empty:
            raise ValueError(f"no training data before {first}")
        # Leakage guard: every training target period has ended by the issue time.
        if train.index.max() + PERIOD > issue_time(first, hour):
            raise AssertionError(f"training data for {first} extends past its issue time")

        model = make_model().fit(train[features], train[TARGET])
        rows = data[data["delivery_day"].isin(block)]
        if getattr(model, "uses_history", False):
            pred = _predict_with_history(model, data, block, features, hour)
        else:
            pred = model.predict_quantiles(rows[features])[list(QUANTILE_COLUMNS)]
        pred["y"] = rows[TARGET]
        pred["delivery_day"] = rows["delivery_day"]
        pred["trained_through"] = first - dt.timedelta(days=2)
        outputs.append(pred)
    return pd.concat(outputs)


def _config(obj: object) -> str:
    """Deterministic text of an unfitted model's settings (no memory addresses)."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_config(o) for o in obj) + "]"
    if isinstance(obj, dict):
        return "{" + ",".join(f"{k}:{_config(v)}" for k, v in sorted(obj.items())) + "}"
    if hasattr(obj, "__dict__"):
        return type(obj).__name__ + _config(vars(obj))
    return repr(obj)


def model_fingerprint(make_model: Callable[[], Forecaster]) -> str:
    """Short hash of a model's code and settings, so cached results go stale when either changes.

    Covers the source of the model's module and of every other ``gb_price_sandbox.models``
    module it uses (e.g. QRA depends on LEAR), plus the settings of an unfitted instance
    (so LightGBM light and heavy differ).
    """
    model = make_model()
    module = sys.modules[type(model).__module__]
    names = {module.__name__}
    for value in vars(module).values():
        source_module = getattr(value, "__module__", None)
        if isinstance(source_module, str) and source_module.startswith("gb_price_sandbox.models"):
            names.add(source_module)
    sources = "".join(inspect.getsource(sys.modules[n]) for n in sorted(names))
    return hashlib.sha1((sources + _config(model)).encode()).hexdigest()[:10]


def backtest_cache_path(
    data: pd.DataFrame,
    name: str,
    make_model: Callable[[], Forecaster],
    test_start: dt.date,
    test_end: dt.date,
    settings: Settings,
    *,
    hour: int = 10,
    retrain_every_days: int = 7,
    max_train_days: int | None = None,
):
    """Where :func:`cached_walk_forward` stores (or would store) this backtest."""
    slug = "".join(c if c.isalnum() else "_" for c in name.lower())
    visible = data[data["delivery_day"] < test_end]
    data_hash = hashlib.sha1(pd.util.hash_pandas_object(visible).to_numpy().tobytes()).hexdigest()
    key = (
        f"{slug}_{model_fingerprint(make_model)}_{test_start}_{test_end}_h{hour:02d}"
        f"_r{retrain_every_days}_w{max_train_days or 'all'}_data{data_hash[:10]}"
    )
    return settings.data_dir / "processed" / "backtests" / f"{key}.parquet"


def cached_walk_forward(
    data: pd.DataFrame,
    name: str,
    make_model: Callable[[], Forecaster],
    test_start: dt.date,
    test_end: dt.date,
    settings: Settings,
    *,
    hour: int = 10,
    retrain_every_days: int = 7,
    max_train_days: int | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """:func:`walk_forward`, cached as parquet under ``data/processed/backtests``.

    The cache key covers the model name, its source-code fingerprint, the test window, the
    issue hour and a hash of the data rows the backtest can see (everything before
    ``test_end``), so any of those changing triggers a re-run. Appending newer data does not.
    """
    path = backtest_cache_path(
        data,
        name,
        make_model,
        test_start,
        test_end,
        settings,
        hour=hour,
        retrain_every_days=retrain_every_days,
        max_train_days=max_train_days,
    )
    out = cached_parquet(
        path,
        lambda: walk_forward(
            data,
            make_model,
            test_start,
            test_end,
            hour=hour,
            retrain_every_days=retrain_every_days,
            max_train_days=max_train_days,
        ),
        refresh=refresh,
    )
    out["delivery_day"] = pd.to_datetime(out["delivery_day"]).dt.date
    return out
