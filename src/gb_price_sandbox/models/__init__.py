"""Forecast models and the registry the app and backtests use.

Models are grouped into *families* (the kinds of model the Forecast sandbox explains); a
family can offer more than one fixed configuration, e.g. LightGBM light and heavy.
"""

from collections.abc import Callable
from dataclasses import dataclass

from gb_price_sandbox.models.base import (
    QUANTILE_COLUMNS,
    QUANTILES,
    Forecaster,
    HistoryForecaster,
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
    """

    name: str
    family: str
    factory: Callable[[], Forecaster | HistoryForecaster]
    probabilistic: bool
    summary: str


def _lightgbm(preset: str) -> Callable[[], LightGBMQuantile]:
    def make() -> LightGBMQuantile:
        return LightGBMQuantile(preset)

    make.__name__ = f"lightgbm_{preset}"
    return make


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
    ModelSpec(MSTLAR.name, "MSTL", MSTLAR, True, "Daily and weekly seasonality plus AR(2)"),
    ModelSpec(
        SARIMAXFourier.name,
        "SARIMAX",
        SARIMAXFourier,
        True,
        "ARIMA errors, Fourier seasonality, demand and wind",
    ),
    ModelSpec(
        LinearQuantileRegression.name,
        "Quantile regression",
        LinearQuantileRegression,
        True,
        "LEAR's inputs, fitted for each quantile",
    ),
    ModelSpec(QRA.name, "QRA", QRA, True, "Quantile regression on the point models' forecasts"),
    ModelSpec("LightGBM (light)", "LightGBM", _lightgbm("light"), True, "200 shallow trees"),
    ModelSpec("LightGBM (heavy)", "LightGBM", _lightgbm("heavy"), True, "500 deeper trees"),
    ModelSpec(
        QuantileForest.name,
        "Quantile forest",
        QuantileForest,
        True,
        "Random forest reading quantiles off its leaves",
    ),
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
    "WeekdayProfile",
    "point_as_quantiles",
    "quantile_column",
    "sort_quantiles",
]
