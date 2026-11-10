"""Walk-forward backtest: expanding window, issued D-1 for delivery day D, weekly retrain."""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
from collections.abc import Callable

import pandas as pd

from gb_price_sandbox.config import Settings
from gb_price_sandbox.data.cache import cached_parquet
from gb_price_sandbox.features.dataset import FEATURE_COLUMNS, TARGET
from gb_price_sandbox.models.base import QUANTILE_COLUMNS, Forecaster
from gb_price_sandbox.timing import PERIOD, issue_time


def training_mask(data: pd.DataFrame, first_forecast_day: dt.date) -> pd.Series:
    """Rows usable for a model issued at the cutoff of ``first_forecast_day`` (D-1 morning).

    Only delivery days up to D-2 have fully settled prices by then, so those are the
    training targets. Rows with a missing target are dropped.
    """
    last_train_day = first_forecast_day - dt.timedelta(days=2)
    return (data["delivery_day"] <= last_train_day) & data[TARGET].notna()


def walk_forward(
    data: pd.DataFrame,
    make_model: Callable[[], Forecaster],
    test_start: dt.date,
    test_end: dt.date,
    *,
    retrain_every_days: int = 7,
    hour: int = 10,
) -> pd.DataFrame:
    """Backtest a model over delivery days ``[test_start, test_end)``.

    Every ``retrain_every_days`` days a fresh model is fitted on all rows allowed by
    :func:`training_mask` (an expanding window). It then forecasts the next block of days.
    The model sees only :data:`FEATURE_COLUMNS`.

    Args:
        data: Output of :func:`gb_price_sandbox.features.dataset.build_dataset`.
        make_model: Zero-argument factory returning an unfitted model.
        test_start: First delivery day forecast (inclusive).
        test_end: Last delivery day forecast (exclusive).
        retrain_every_days: Retrain cadence in days.
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
        train = data[training_mask(data, first)]
        if train.empty:
            raise ValueError(f"no training data before {first}")
        # Leakage guard: every training target period has ended by the issue time.
        if train.index.max() + PERIOD > issue_time(first, hour):
            raise AssertionError(f"training data for {first} extends past its issue time")

        model = make_model().fit(train[features], train[TARGET])
        rows = data[data["delivery_day"].isin(block)]
        pred = model.predict_quantiles(rows[features])[list(QUANTILE_COLUMNS)]
        pred["y"] = rows[TARGET]
        pred["delivery_day"] = rows["delivery_day"]
        pred["trained_through"] = first - dt.timedelta(days=2)
        outputs.append(pred)
    return pd.concat(outputs)


def model_fingerprint(make_model: Callable[[], Forecaster]) -> str:
    """Short hash of a model's source code, so cached results go stale when the code changes."""
    model_cls = make_model if isinstance(make_model, type) else type(make_model())
    source = inspect.getsource(model_cls)
    return hashlib.sha1(source.encode()).hexdigest()[:10]


def cached_walk_forward(
    data: pd.DataFrame,
    name: str,
    make_model: Callable[[], Forecaster],
    test_start: dt.date,
    test_end: dt.date,
    settings: Settings,
    *,
    hour: int = 10,
    refresh: bool = False,
) -> pd.DataFrame:
    """:func:`walk_forward`, cached as parquet under ``data/processed/backtests``.

    The cache key covers the model name, its source-code fingerprint, the test window, the
    issue hour and a hash of the data rows the backtest can see (everything before
    ``test_end``), so any of those changing triggers a re-run. Appending newer data does not.
    """
    slug = "".join(c if c.isalnum() else "_" for c in name.lower())
    visible = data[data["delivery_day"] < test_end]
    data_hash = hashlib.sha1(pd.util.hash_pandas_object(visible).to_numpy().tobytes()).hexdigest()
    key = (
        f"{slug}_{model_fingerprint(make_model)}_{test_start}_{test_end}_h{hour:02d}"
        f"_data{data_hash[:10]}"
    )
    path = settings.data_dir / "processed" / "backtests" / f"{key}.parquet"
    out = cached_parquet(
        path,
        lambda: walk_forward(data, make_model, test_start, test_end, hour=hour),
        refresh=refresh,
    )
    out["delivery_day"] = pd.to_datetime(out["delivery_day"]).dt.date
    return out
