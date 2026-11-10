"""Forecast models: seasonal naive, LEAR, linear quantile regression, LightGBM quantile."""

from collections.abc import Callable

from gb_price_sandbox.models.base import (
    QUANTILE_COLUMNS,
    QUANTILES,
    Forecaster,
    point_as_quantiles,
    quantile_column,
)
from gb_price_sandbox.models.baselines import LEAR, SeasonalNaive

#: Model registry: display name -> zero-argument factory. Order sets the display order and
#: each model's fixed chart colour.
MODELS: dict[str, Callable[[], Forecaster]] = {
    SeasonalNaive.name: SeasonalNaive,
    LEAR.name: LEAR,
}

__all__ = [
    "LEAR",
    "MODELS",
    "QUANTILES",
    "QUANTILE_COLUMNS",
    "Forecaster",
    "SeasonalNaive",
    "point_as_quantiles",
    "quantile_column",
]
