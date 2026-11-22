"""Tree ensembles for quantile forecasting: LightGBM and a quantile regression forest."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

from gb_price_sandbox.features.dataset import FEATURE_COLUMNS
from gb_price_sandbox.models.base import QUANTILE_COLUMNS, QUANTILES, sort_quantiles


@dataclass(frozen=True)
class LightGBMPreset:
    """A fixed LightGBM configuration (no tuning: the presets are the choice)."""

    n_estimators: int
    num_leaves: int
    learning_rate: float
    min_child_samples: int
    subsample: float = 1.0
    colsample_bytree: float = 1.0


LIGHTGBM_PRESETS: dict[str, LightGBMPreset] = {
    # Few, shallow trees: quick to refit, hard to overfit.
    "light": LightGBMPreset(
        n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=50
    ),
    # More and deeper trees, with row and column subsampling to keep it from memorising.
    "heavy": LightGBMPreset(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
    ),
}


class LightGBMQuantile:
    """Gradient-boosted trees fitted with the pinball loss, one model per quantile.

    Boosting adds small trees one at a time, each fitted to the errors (strictly, the
    gradient of the loss) of the ensemble so far. With the pinball loss at q = 0.1, 0.5 and
    0.9 the three ensembles give a median and an 80% band. A single model covers all 48
    half-hours, with the half-hour and day of week as inputs, so trees can find interactions
    such as "wind matters more overnight" by themselves. It sees every column in
    :data:`FEATURE_COLUMNS`, including residual demand. Missing inputs are handled natively.
    """

    def __init__(self, preset: str = "light") -> None:
        """Pick a configuration from :data:`LIGHTGBM_PRESETS`."""
        self.preset = preset
        self.params = asdict(LIGHTGBM_PRESETS[preset])
        self.name = f"LightGBM ({preset})"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMQuantile:  # noqa: N803
        """Fit one booster per quantile."""
        ok = y.notna()
        features = X.loc[ok, list(FEATURE_COLUMNS)].astype(float)
        self.models_ = [
            lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                subsample_freq=1 if self.params["subsample"] < 1 else 0,
                random_state=0,
                verbose=-1,
                **self.params,
            ).fit(features, y[ok])
            for q in QUANTILES
        ]
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Quantile forecasts, sorted so they never cross."""
        features = X[list(FEATURE_COLUMNS)].astype(float)
        values = np.column_stack([m.predict(features) for m in self.models_])
        return sort_quantiles(pd.DataFrame(values, index=X.index, columns=list(QUANTILE_COLUMNS)))


class QuantileForest:
    """Quantile regression forest (Meinshausen, 2006).

    A random forest grows many deep trees, each on a random sample of rows, and averages them.
    A *quantile* forest keeps every training price that lands in each leaf. To forecast, it
    pools the training prices sharing a leaf with the new row across all trees and reads
    the 10th, 50th and 90th percentiles off that pool. One forest gives every quantile, and
    the bands come from the spread of similar past situations rather than an assumed shape.

    Settings are fixed: 100 trees, at least 50 rows per leaf, each tree on a 30% sample.
    Rows with a missing input are dropped from training and get no forecast.
    """

    name = "Quantile regression forest"

    def __init__(self) -> None:
        """Fixed settings (see the class docstring)."""
        self.params = {"n_estimators": 100, "min_samples_leaf": 50, "max_samples": 0.3}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileForest:  # noqa: N803
        """Grow the forest on complete rows."""
        features = X[list(FEATURE_COLUMNS)].astype(float)
        ok = features.notna().all(axis=1) & y.notna()
        self.model_ = RandomForestQuantileRegressor(random_state=0, n_jobs=-1, **self.params).fit(
            features[ok], y[ok]
        )
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Quantiles from the pooled leaf samples; NaN where inputs are missing."""
        features = X[list(FEATURE_COLUMNS)].astype(float)
        ok = features.notna().all(axis=1).to_numpy()
        out = pd.DataFrame(np.nan, index=X.index, columns=list(QUANTILE_COLUMNS))
        if ok.any():
            values = self.model_.predict(features[ok], quantiles=list(QUANTILES))
            out.loc[ok] = np.asarray(values, dtype=float)
        return sort_quantiles(out)
