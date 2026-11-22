"""Classical time-series models: MSTL + AR, and SARIMAX with Fourier and exogenous terms.

Unlike the other models these work on the price *series* itself. Each is refitted weekly on a
trailing window, then for every delivery day it is updated with the prices published by the
D-1 issue time (without re-estimating its parameters) and forecasts forward from there: through
the rest of D-1 and across all of D.

Both work on LEAR's asinh price scale and transform their forecasts back. Their bands are the
model's own Gaussian prediction intervals on that scale, which become skewed bands in £/MWh.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.statespace.sarimax import SARIMAX

from gb_price_sandbox.models.base import QUANTILE_COLUMNS, Objective
from gb_price_sandbox.timing import LONDON, PERIOD

#: Trailing window (days) each refit and each daily update uses, unless a model overrides it.
WINDOW_DAYS = 28
#: Two-sided level of the prediction interval that gives q10 and q90.
_ALPHA = 0.2


def _regular(series: pd.Series, days: int = WINDOW_DAYS) -> pd.Series:
    """The last ``days`` days of ``series`` on a complete half-hourly grid."""
    end = series.index.max()
    start = end - pd.Timedelta(days=days) + PERIOD
    grid = pd.date_range(start, end, freq=PERIOD)
    return series.reindex(grid)


def _robust_scale(values: pd.Series) -> tuple[float, float]:
    v = values.dropna()
    median = float(np.median(v))
    mad = float(np.median(np.abs(v - median))) or 1.0
    return median, mad


def _empty(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=list(QUANTILE_COLUMNS))


def _bands(lower: np.ndarray, mean: np.ndarray, upper: np.ndarray, objective: Objective):
    """Stack (q10, q50, q90); with the mean objective, report the mean in every column."""
    if objective == "mean":
        return np.column_stack([mean, mean, mean])
    return np.column_stack([lower, mean, upper])


class MSTLAR:
    """MSTL decomposition plus an AR(2) model for what's left.

    MSTL (multiple seasonal-trend decomposition by Loess; Bandara et al., 2021) splits the
    asinh price into a **daily** seasonal pattern (48 periods), a **weekly** one (336), a
    slowly moving trend and a remainder. The seasonal patterns are carried forward by
    repeating their latest cycle. Trend plus remainder is forecast with an AR(2) model, which
    pulls the latest deviation from the usual pattern gradually back to its mean. The
    seasonal phase is counted in UTC half-hours, so the daily pattern slips by an hour for a
    few weeks around each clock change.
    """

    name = "MSTL + AR"
    uses_history = True

    def __init__(self, objective: Objective = "quantile") -> None:
        """Fixed settings: 28-day window, periods 48 and 336, AR(2) with a constant."""
        self.objective = objective
        self.periods = (48, 336)
        self.order = (2, 0, 0)

    def _season(self, times: pd.DatetimeIndex) -> np.ndarray:
        pos = ((times - self.start_) / PERIOD).to_numpy().astype(int)
        total = np.zeros(len(times))
        for last, p in zip(self.last_cycles_, self.periods, strict=True):
            total += last[(pos - (self.n_ - p)) % p]
        return total

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MSTLAR:  # noqa: N803
        """Decompose the last 28 days and fit the AR model to trend plus remainder."""
        window = _regular(y)
        self.median_, self.mad_ = _robust_scale(window)
        z = np.arcsinh((window - self.median_) / self.mad_).interpolate(limit_direction="both")
        self.start_, self.n_ = window.index[0], len(window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on short windows
            res = MSTL(z.to_numpy(), periods=self.periods).fit()
            seasonal = np.asarray(res.seasonal)
            self.last_cycles_ = [seasonal[-p:, k] for k, p in enumerate(self.periods)]
            deseason = z.to_numpy() - self._season(window.index)
            self.arima_ = ARIMA(deseason, order=self.order, trend="c").fit()
        return self

    def predict_day(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """Update the AR model on the latest 28 days of published prices and forecast."""
        window = _regular(history["price"])
        z = np.arcsinh((window - self.median_) / self.mad_).interpolate(limit_direction="both")
        if z.isna().all():
            return _empty(future.index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on short windows
            updated = self.arima_.apply(z.to_numpy() - self._season(window.index))
            fc = updated.get_forecast(len(future))
            mean = np.asarray(fc.predicted_mean)
            ci = np.asarray(fc.conf_int(alpha=_ALPHA))
        season = self._season(future.index)
        zq = _bands(ci[:, 0], mean, ci[:, 1], self.objective) + season[:, None]
        values = np.sinh(zq) * self.mad_ + self.median_
        return pd.DataFrame(values, index=future.index, columns=list(QUANTILE_COLUMNS))


EXOG_COLUMNS = ("demand_mw", "wind_mw")


class SARIMAXFourier:
    """Regression with ARIMA errors: Fourier seasonality plus demand and wind forecasts.

    A seasonal ARIMA with a 48- and 336-period season is far too slow to fit on half-hourly
    data, so the seasons go into the regression instead, as **Fourier terms**: sine and cosine
    waves at 1, 2 and 3 cycles per day and 1 and 2 cycles per week (local clock time, so clock
    changes are handled). NESO's demand and wind forecasts are added as further regressors.
    The errors follow an ARMA(1, 1), which captures how today's surprise carries into
    tomorrow. The result is the SARIMAX model class with all the seasonality in the X.
    """

    name = "SARIMAX"
    uses_history = True

    def __init__(self, objective: Objective = "quantile") -> None:
        """Fixed settings: 14-day window, ARMA(1, 1) errors, 3 daily and 2 weekly harmonics.

        Kept small because estimating the parameters dominates its run time.
        """
        self.objective = objective
        self.window_days = 14
        self.order = (1, 0, 1)
        self.daily_harmonics = 3
        self.weekly_harmonics = 2

    def _exog(self, frame: pd.DataFrame) -> np.ndarray:
        local = frame.index.tz_convert(LONDON)
        minutes = (local.hour * 60 + local.minute).to_numpy()
        day = minutes / 1440
        week = (local.day_of_week.to_numpy() * 1440 + minutes) / (7 * 1440)
        cols = []
        for k in range(1, self.daily_harmonics + 1):
            cols += [np.sin(2 * np.pi * k * day), np.cos(2 * np.pi * k * day)]
        for k in range(1, self.weekly_harmonics + 1):
            cols += [np.sin(2 * np.pi * k * week), np.cos(2 * np.pi * k * week)]
        exog = frame[list(EXOG_COLUMNS)].astype(float).interpolate(limit_direction="both")
        scaled = ((exog - self.exog_mean_) / self.exog_std_).fillna(0.0).to_numpy()
        # The intercept is an explicit regressor (rather than trend="c"), so a forecast column
        # that happens to be constant over the window can't clash with it.
        return np.column_stack([np.ones(len(frame)), *cols, scaled])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SARIMAXFourier:  # noqa: N803
        """Fit on the last 14 days (missing prices are skipped by the Kalman filter)."""
        window = _regular(y, self.window_days)
        features = X[list(EXOG_COLUMNS)].reindex(window.index)
        self.median_, self.mad_ = _robust_scale(window)
        self.exog_mean_ = features.mean()
        self.exog_std_ = features.std().replace(0, 1.0)
        z = np.arcsinh((window - self.median_) / self.mad_).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on short windows
            self.model_ = SARIMAX(z, exog=self._exog(features), order=self.order, trend="n").fit(
                disp=False, maxiter=100
            )
        return self

    def predict_day(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """Re-run the filter on the latest 14 days (same parameters) and forecast."""
        window = _regular(history["price"], self.window_days)
        hist = history.reindex(window.index)
        z = np.arcsinh((window - self.median_) / self.mad_).to_numpy()
        if np.isnan(z).all():
            return _empty(future.index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on short windows
            updated = self.model_.apply(z, exog=self._exog(hist))
            fc = updated.get_forecast(len(future), exog=self._exog(future))
            mean = np.asarray(fc.predicted_mean)
            ci = np.asarray(fc.conf_int(alpha=_ALPHA))
        zq = _bands(ci[:, 0], mean, ci[:, 1], self.objective)
        values = np.sinh(zq) * self.mad_ + self.median_
        return pd.DataFrame(values, index=future.index, columns=list(QUANTILE_COLUMNS))
