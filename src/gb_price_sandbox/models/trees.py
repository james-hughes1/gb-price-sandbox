"""Tree ensembles for quantile forecasting: LightGBM and a quantile regression forest."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

from gb_price_sandbox.features.sets import CORE, FeatureSet
from gb_price_sandbox.models.base import (
    QUANTILE_COLUMNS,
    QUANTILES,
    Objective,
    point_as_quantiles,
    sort_quantiles,
)

#: Fixed LightGBM settings, kept deliberately small so a 4-year walk-forward stays quick:
#: 100 shallow trees with a high learning rate, each fitted on half the rows.
LIGHTGBM_PARAMS: dict[str, float | int] = {
    "n_estimators": 100,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "min_child_samples": 100,
    "subsample": 0.5,
    "subsample_freq": 1,
    "max_bin": 63,
}


class LightGBMQuantile:
    """Gradient-boosted trees fitted with the pinball loss, one model per quantile.

    Boosting adds small trees one at a time, each fitted to the errors (strictly, the
    gradient of the loss) of the ensemble so far. With the pinball loss at q = 0.1, 0.5 and
    0.9 the three ensembles give a median and an 80% band. A single model covers all 48
    half-hours, with the half-hour and day of week as inputs, so trees can find interactions
    such as "wind matters more overnight" by themselves. It sees every column of its feature
    set, including residual demand. Missing inputs are handled natively. With the ``mean``
    objective a single booster is fitted with squared error instead.
    Settings are fixed (:data:`LIGHTGBM_PARAMS`) and small, for speed.
    """

    name = "LightGBM"

    def __init__(self, features: FeatureSet = CORE, objective: Objective = "quantile") -> None:
        """Fixed settings (see :data:`LIGHTGBM_PARAMS`); feature set and objective."""
        self.features = features
        self.objective = objective
        self.params = dict(LIGHTGBM_PARAMS)

    def _boosters(self) -> list[tuple[str, float | None]]:
        if self.objective == "mean":
            return [("regression", None)]  # squared error: the conditional mean
        return [("quantile", q) for q in QUANTILES]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMQuantile:  # noqa: N803
        """Fit one booster per quantile (or one for the mean)."""
        ok = y.notna()
        features = X.loc[ok, list(self.features.columns)].astype(float)
        self.models_ = []
        for objective, alpha in self._boosters():
            extra = {"alpha": alpha} if alpha is not None else {}
            booster = lgb.LGBMRegressor(
                objective=objective, random_state=0, verbose=-1, **extra, **self.params
            )
            self.models_.append(booster.fit(features, y[ok]))
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Quantile forecasts, sorted so they never cross (or the mean in every column)."""
        features = X[list(self.features.columns)].astype(float)
        values = np.column_stack([m.predict(features) for m in self.models_])
        if values.shape[1] == 1:
            return point_as_quantiles(pd.Series(values[:, 0], index=X.index))
        return sort_quantiles(pd.DataFrame(values, index=X.index, columns=list(QUANTILE_COLUMNS)))


class QuantileForest:
    """Quantile regression forest (Meinshausen, 2006).

    A random forest grows many deep trees, each on a random sample of rows, and averages them.
    A *quantile* forest keeps every training price that lands in each leaf. To forecast, it
    pools the training prices sharing a leaf with the new row across all trees and reads
    the 10th, 50th and 90th percentiles off that pool. One forest gives every quantile, and
    the bands come from the spread of similar past situations rather than an assumed shape.

    Settings are fixed and small, for speed: 50 trees, at least 50 rows per leaf, each tree
    on a 20% sample. Rows with a missing input are dropped from training and get no
    forecast. With the ``mean`` objective it reports the forest's average instead.
    """

    name = "Quantile regression forest"

    def __init__(self, features: FeatureSet = CORE, objective: Objective = "quantile") -> None:
        """Fixed settings (see the class docstring); feature set and objective."""
        self.features = features
        self.objective = objective
        self.params = {"n_estimators": 50, "min_samples_leaf": 50, "max_samples": 0.2}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileForest:  # noqa: N803
        """Grow the forest on complete rows."""
        features = X[list(self.features.columns)].astype(float)
        ok = features.notna().all(axis=1) & y.notna()
        self.model_ = RandomForestQuantileRegressor(random_state=0, n_jobs=-1, **self.params).fit(
            features[ok], y[ok]
        )
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Quantiles from the pooled leaf samples; NaN where inputs are missing."""
        features = X[list(self.features.columns)].astype(float)
        ok = features.notna().all(axis=1).to_numpy()
        if self.objective == "mean":
            point = pd.Series(np.nan, index=X.index)
            if ok.any():
                point[ok] = np.asarray(self.model_.predict(features[ok], quantiles="mean"))
            return point_as_quantiles(point)
        out = pd.DataFrame(np.nan, index=X.index, columns=list(QUANTILE_COLUMNS))
        if ok.any():
            values = self.model_.predict(features[ok], quantiles=list(QUANTILES))
            out.loc[ok] = np.asarray(values, dtype=float)
        return sort_quantiles(out)
