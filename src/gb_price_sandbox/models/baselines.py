"""Baseline forecasters: naive benchmarks and LEAR."""

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


class D2Persistence:
    """Predict the price in the same half-hour two days earlier.

    "Tomorrow looks like the most recent day we know about." D-1 isn't fully known at the
    issue time, so D-2 is the latest complete day. It tracks the price level well but ignores
    the weekly cycle (Monday is forecast from Saturday).
    """

    name = "D-2 persistence"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> D2Persistence:  # noqa: N803
        """No parameters to learn."""
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return ``price_lag_2d`` as a point forecast."""
        return point_as_quantiles(X["price_lag_2d"])


WEEKDAY_LAGS = ("price_lag_7d", "price_lag_14d", "price_lag_21d", "price_lag_28d")


class WeekdayProfile:
    """Predict the average of the same half-hour on the last four same weekdays.

    A smoother version of the seasonal naive: averaging four Mondays damps the effect of one
    unusually windy or still Monday, at the cost of reacting more slowly to changes in level.
    """

    name = "Weekday profile"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WeekdayProfile:  # noqa: N803
        """No parameters to learn."""
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Mean of the 7, 14, 21 and 28-day lags (NaN only if all four are missing)."""
        lags = X[list(WEEKDAY_LAGS)]
        return point_as_quantiles(lags.mean(axis=1, skipna=True))


PRICE_FEATURES = ("price_lag_2d", "price_lag_3d", "price_lag_7d", "price_mean_d2")
EXOG_FEATURES = ("demand_mw", "wind_mw")


class AsinhDesign:
    """LEAR's input transforms, shared with the linear quantile models.

    - Prices (target and price inputs) -> ``asinh((p - median) / MAD)``, with median and MAD
      from the training target. Tames spikes and copes with negative prices.
    - Exogenous forecasts -> standardised with training mean and standard deviation.
    - Day of week -> six 0/1 dummies (Monday is the baseline).
    """

    def __init__(
        self,
        price_features: tuple[str, ...] = PRICE_FEATURES,
        exog_features: tuple[str, ...] = EXOG_FEATURES,
    ):
        """Configure the input columns."""
        self.price_features = price_features
        self.exog_features = exog_features

    @property
    def columns(self) -> list[str]:
        """Every input column the design reads."""
        return [*self.price_features, *self.exog_features, "day_of_week", "period_of_day"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AsinhDesign:  # noqa: N803
        """Learn the centring and scaling from the training rows."""
        self.median_ = float(np.median(y))
        self.mad_ = float(np.median(np.abs(y - self.median_))) or 1.0
        exog = X[list(self.exog_features)].to_numpy(dtype=float)
        self.exog_mean_ = exog.mean(axis=0)
        self.exog_std_ = exog.std(axis=0) + 1e-9
        return self

    def to_z(self, prices: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """Prices to the asinh scale."""
        return np.arcsinh((np.asarray(prices, dtype=float) - self.median_) / self.mad_)

    def from_z(self, z: np.ndarray) -> np.ndarray:
        """Convert from the asinh scale back to prices."""
        return np.sinh(z) * self.mad_ + self.median_

    def matrix(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Design matrix (no intercept column)."""
        price_part = self.to_z(X[list(self.price_features)])
        exog = X[list(self.exog_features)].to_numpy(dtype=float)
        exog_part = (exog - self.exog_mean_) / self.exog_std_
        dow = X["day_of_week"].to_numpy()
        dow_part = (dow[:, None] == np.arange(1, 7)[None, :]).astype(float)
        return np.hstack([price_part, exog_part, dow_part])

    def complete(self, X: pd.DataFrame) -> pd.Series:  # noqa: N803
        """Rows with every input present."""
        return X[self.columns].notna().all(axis=1)


class LEAR:
    """LASSO-estimated autoregressive model with exogenous inputs (Lago et al., 2021).

    A simplified version of the standard EPF benchmark:

    - One LASSO model per local half-hour of the day (48 models), so each period gets its
      own coefficients.
    - Inputs and target go through :class:`AsinhDesign` (asinh prices, standardised demand
      and wind, day-of-week dummies).
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
        self.design = AsinhDesign(price_features, exog_features)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LEAR:  # noqa: N803
        """Fit one LASSO per period of day on rows with complete inputs."""
        ok = self.design.complete(X) & y.notna()
        X, y = X[ok], y[ok]  # noqa: N806
        self.design.fit(X, y)
        matrix = self.design.matrix(X)
        target = self.design.to_z(y)
        periods = X["period_of_day"].to_numpy()
        self.models_: dict[int, LassoLarsIC] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for p in np.unique(periods):
                rows = periods == p
                self.models_[int(p)] = LassoLarsIC(criterion="aic").fit(matrix[rows], target[rows])
        return self

    def predict_point(self, X: pd.DataFrame) -> pd.Series:  # noqa: N803
        """Point forecast in £/MWh; NaN where inputs are missing."""
        out = pd.Series(np.nan, index=X.index)
        ok = self.design.complete(X).to_numpy()
        if ok.any():
            matrix = self.design.matrix(X[ok])
            periods = X.loc[ok, "period_of_day"].to_numpy()
            z = np.full(len(matrix), np.nan)
            for p in np.unique(periods):
                rows = periods == p
                model = self.models_.get(int(p))
                if model is not None:
                    z[rows] = model.predict(matrix[rows])
            out[ok] = self.design.from_z(z)
        return out

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Point forecast (as degenerate quantiles); NaN where inputs are missing."""
        return point_as_quantiles(self.predict_point(X))
