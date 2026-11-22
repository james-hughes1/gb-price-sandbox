"""Forecast models and the registry the app and backtests use.

Models are grouped into *families* (the kinds of model the Forecast sandbox explains); a
family can offer more than one fixed configuration, e.g. LightGBM light and heavy.
"""

from collections.abc import Callable
from dataclasses import dataclass

from gb_price_sandbox.features.sets import CORE, FeatureSet
from gb_price_sandbox.models.base import (
    QUANTILE_COLUMNS,
    QUANTILES,
    Forecaster,
    HistoryForecaster,
    Objective,
    point_as_quantiles,
    quantile_column,
    sort_quantiles,
)
from gb_price_sandbox.models.baselines import LEAR, D2Persistence, SeasonalNaive, WeekdayProfile
from gb_price_sandbox.models.linear_quantile import QRA, LinearQuantileRegression
from gb_price_sandbox.models.timeseries import MSTLAR, SARIMAXFourier
from gb_price_sandbox.models.trees import LightGBMQuantile, QuantileForest


@dataclass(frozen=True)
class ModelSpec:
    """A model on offer.

    Attributes:
        name: Display name (unique; also the cache key for backtests).
        family: The kind of model, for grouping and explanations.
        factory: Zero-argument callable returning an unfitted model.
        probabilistic: True if it forecasts a band (q10 to q90), not just a point.
        summary: One line for selectors and tables.
        retrain_every_days: Walk-forward refit interval. Weekly by default; the slowest
            models refit every 4 weeks so a full backtest takes minutes, not hours.
        max_train_days: Trailing training window in days (None: all history). Used by the
            slowest models, again for speed.
        options: Which sandbox options the model responds to: ``"features"`` (it can use
            extra feature groups) and/or ``"objective"`` (it can fit quantiles or the mean).
    """

    name: str
    family: str
    factory: Callable[[], Forecaster | HistoryForecaster]
    probabilistic: bool
    summary: str
    retrain_every_days: int = 7
    max_train_days: int | None = None
    options: frozenset[str] = frozenset()


#: Refit interval for the slowest models.
SLOW_REFIT_DAYS = 28


#: Every model, in display order. Families are contiguous.
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        SeasonalNaive.name, "Naive baselines", SeasonalNaive, False, "Same half-hour last week"
    ),
    ModelSpec(D2Persistence.name, "Naive baselines", D2Persistence, False, "Same half-hour on D-2"),
    ModelSpec(
        WeekdayProfile.name,
        "Naive baselines",
        WeekdayProfile,
        False,
        "Average of the last four same weekdays",
    ),
    ModelSpec(LEAR.name, "LEAR", LEAR, False, "LASSO on lagged prices, demand and wind"),
    # The two time-series models update on the latest prices every day without refitting,
    # so their parameters only need re-estimating every 4 weeks.
    ModelSpec(
        MSTLAR.name,
        "MSTL",
        MSTLAR,
        True,
        "Daily and weekly seasonality plus AR(2)",
        SLOW_REFIT_DAYS,
    ),
    ModelSpec(
        SARIMAXFourier.name,
        "SARIMAX",
        SARIMAXFourier,
        True,
        "ARIMA errors, Fourier seasonality, demand and wind",
        SLOW_REFIT_DAYS,
    ),
    ModelSpec(
        LinearQuantileRegression.name,
        "Quantile regression",
        LinearQuantileRegression,
        True,
        "LEAR's inputs, fitted for each quantile",
        # 144 separate fits per refit (48 half-hours x 3 quantiles), so it refits quarterly.
        91,
        max_train_days=365,
    ),
    ModelSpec(
        QRA.name,
        "QRA",
        QRA,
        True,
        "Quantile regression on the point models' forecasts",
        SLOW_REFIT_DAYS,
        max_train_days=730,
    ),
    ModelSpec(
        LightGBMQuantile.name,
        "LightGBM",
        LightGBMQuantile,
        True,
        "Gradient-boosted trees, 100 small trees per quantile",
        SLOW_REFIT_DAYS,
    ),
    ModelSpec(
        QuantileForest.name,
        "Quantile forest",
        QuantileForest,
        True,
        "Random forest reading quantiles off its leaves",
        SLOW_REFIT_DAYS,
        max_train_days=730,
    ),
)

_OPTIONS: dict[str, frozenset[str]] = {
    LEAR.name: frozenset({"features"}),
    MSTLAR.name: frozenset({"objective"}),
    SARIMAXFourier.name: frozenset({"objective"}),
    LinearQuantileRegression.name: frozenset({"features", "objective"}),
    QRA.name: frozenset({"features", "objective"}),
    LightGBMQuantile.name: frozenset({"features", "objective"}),
    QuantileForest.name: frozenset({"features", "objective"}),
}
MODEL_SPECS = tuple(
    ModelSpec(**{**vars(s), "options": _OPTIONS.get(s.name, frozenset())}) for s in MODEL_SPECS
)

#: Model registry: display name -> zero-argument factory. Order sets the display order.
MODELS: dict[str, Callable[[], Forecaster | HistoryForecaster]] = {
    s.name: s.factory for s in MODEL_SPECS
}
MODEL_SPECS_BY_NAME: dict[str, ModelSpec] = {s.name: s for s in MODEL_SPECS}
#: Families in display order.
FAMILIES: tuple[str, ...] = tuple(dict.fromkeys(s.family for s in MODEL_SPECS))
#: Quick to backtest (seconds), so pages can run them without waiting.
DEFAULT_MODELS: tuple[str, ...] = (
    SeasonalNaive.name,
    D2Persistence.name,
    WeekdayProfile.name,
    LEAR.name,
)


def build_model(
    name: str, features: FeatureSet = CORE, objective: Objective = "quantile"
) -> Forecaster | HistoryForecaster:
    """An unfitted model, configured with the options it supports (others are ignored)."""
    spec = MODEL_SPECS_BY_NAME[name]
    kwargs: dict[str, object] = {}
    if "features" in spec.options:
        kwargs["features"] = features
    if "objective" in spec.options:
        kwargs["objective"] = objective
    return spec.factory(**kwargs)


__all__ = [
    "DEFAULT_MODELS",
    "FAMILIES",
    "LEAR",
    "MODELS",
    "MODEL_SPECS",
    "MODEL_SPECS_BY_NAME",
    "MSTLAR",
    "QRA",
    "QUANTILES",
    "QUANTILE_COLUMNS",
    "D2Persistence",
    "Forecaster",
    "HistoryForecaster",
    "LightGBMQuantile",
    "LinearQuantileRegression",
    "ModelSpec",
    "QuantileForest",
    "SARIMAXFourier",
    "SeasonalNaive",
    "build_model",
    "WeekdayProfile",
    "point_as_quantiles",
    "quantile_column",
    "sort_quantiles",
]
