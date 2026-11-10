"""Baseline forecasters: seasonal naive and LEAR."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsIC

from gb_price_sandbox.models.base import point_as_quantiles


class SeasonalNaive:
    """Predict the price in the same half-hour one week earlier.

    The simplest forecast that captures both the daily and the weekly cycle. Any model worth
    using has to beat it.
    """

    name = "Seasonal naive"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaive:  # noqa: N803
        """No parameters to learn."""
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return ``price_lag_7d`` as a point forecast."""
        return point_as_quantiles(X["price_lag_7d"])


PRICE_FEATURES = ("price_lag_2d", "price_lag_3d", "price_lag_7d", "price_mean_d2")
EXOG_FEATURES = ("demand_mw", "wind_mw")


class LEAR:
    """LASSO-estimated autoregressive model with exogenous inputs (Lago et al., 2021).

    A simplified version of the standard EPF benchmark:

    - One LASSO model per local half-hour of the day (48 models), so each period gets its
      own coefficients.
    - Prices (the target and the price lags) go through a variance-stabilising transform:
      ``asinh((p - median) / MAD)``, with median and MAD from the training data. This
      tames spikes and copes with negative prices.
    - Exogenous inputs (demand and wind forecasts) are standardised. Day-of-week dummies are
      added.
    - LASSO strength is chosen by AIC (``LassoLarsIC``), which is fast enough to refit weekly.

    The original LEAR uses the full 24-hour price profile of D-1, D-2, D-3 and D-7 as inputs.
    Here D-1 is not available (GB MID settles intraday, after our D-1 morning cutoff), and only
    same-period lags plus the D-2 daily mean are used.
    """

    name = "LEAR"

    def __init__(
        self,
        price_features: tuple[str, ...] = PRICE_FEATURES,
        exog_features: tuple[str, ...] = EXOG_FEATURES,
    ):
        """Configure the input columns."""
        self.price_features = price_features
        self.exog_features = exog_features

    # -- transforms ------------------------------------------------------------------------

    def _to_z(self, prices: pd.DataFrame | pd.Series) -> np.ndarray:
        return np.arcsinh((np.asarray(prices, dtype=float) - self.median_) / self.mad_)

    def _from_z(self, z: np.ndarray) -> np.ndarray:
        return np.sinh(z) * self.mad_ + self.median_

    def _design(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        price_part = self._to_z(X[list(self.price_features)])
        exog = X[list(self.exog_features)].to_numpy(dtype=float)
        exog_part = (exog - self.exog_mean_) / self.exog_std_
        dow = X["day_of_week"].to_numpy()
        dow_part = (dow[:, None] == np.arange(1, 7)[None, :]).astype(float)  # Monday is baseline
        return np.hstack([price_part, exog_part, dow_part])

    def _columns(self) -> list[str]:
        return [*self.price_features, *self.exog_features, "day_of_week", "period_of_day"]

    # -- API -------------------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LEAR:  # noqa: N803
        """Fit one LASSO per period of day on rows with complete inputs."""
        ok = X[self._columns()].notna().all(axis=1) & y.notna()
        X, y = X[ok], y[ok]  # noqa: N806
        self.median_ = float(np.median(y))
        self.mad_ = float(np.median(np.abs(y - self.median_))) or 1.0
        exog = X[list(self.exog_features)].to_numpy(dtype=float)
        self.exog_mean_ = exog.mean(axis=0)
        self.exog_std_ = exog.std(axis=0) + 1e-9

        design = self._design(X)
        target = self._to_z(y)
        periods = X["period_of_day"].to_numpy()
        self.models_: dict[int, LassoLarsIC] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for p in np.unique(periods):
                rows = periods == p
                self.models_[int(p)] = LassoLarsIC(criterion="aic").fit(design[rows], target[rows])
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Point forecast (as degenerate quantiles); NaN where inputs are missing."""
        out = pd.Series(np.nan, index=X.index)
        ok = X[self._columns()].notna().all(axis=1).to_numpy()
        if ok.any():
            design = self._design(X[ok])
            periods = X.loc[ok, "period_of_day"].to_numpy()
            z = np.full(len(design), np.nan)
            for p in np.unique(periods):
                rows = periods == p
                model = self.models_.get(int(p))
                if model is not None:
                    z[rows] = model.predict(design[rows])
            out[ok] = self._from_z(z)
        return point_as_quantiles(out)
