"""Linear quantile models: quantile regression and quantile regression averaging (QRA)."""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, IterationLimitWarning

from gb_price_sandbox.models.base import QUANTILE_COLUMNS, QUANTILES, sort_quantiles
from gb_price_sandbox.models.baselines import LEAR, WEEKDAY_LAGS, AsinhDesign
from gb_price_sandbox.timing import delivery_day


def _quantreg(y: np.ndarray, X: np.ndarray, q: float) -> np.ndarray:  # noqa: N803
    """Coefficients of a linear quantile regression (``X`` includes the intercept column)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IterationLimitWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        return sm.QuantReg(y, X).fit(q=q, max_iter=2000).params


def _with_intercept(matrix: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(matrix), 1)), matrix])


class LinearQuantileRegression:
    """Linear quantile regression with LEAR's inputs, one model per half-hour and quantile.

    Ordinary regression fits the conditional *mean* by minimising squared error. Quantile
    regression fits a conditional *quantile* by minimising pinball loss, so fitting it at
    q = 0.1, 0.5 and 0.9 gives a median forecast and an 80% band. Inputs and target are on
    LEAR's asinh scale; because quantiles survive monotone transforms, transforming the fitted
    quantiles back gives quantiles of the price. No penalty is applied (unlike LEAR's LASSO).
    """

    name = "Linear quantile regression"

    def __init__(self) -> None:
        """LEAR's default inputs."""
        self.design = AsinhDesign()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LinearQuantileRegression:  # noqa: N803
        """Fit 48 x 3 quantile regressions on rows with complete inputs."""
        ok = self.design.complete(X) & y.notna()
        X, y = X[ok], y[ok]  # noqa: N806
        self.design.fit(X, y)
        matrix = _with_intercept(self.design.matrix(X))
        target = self.design.to_z(y)
        periods = X["period_of_day"].to_numpy()
        self.coefs_: dict[int, np.ndarray] = {}
        for p in np.unique(periods):
            rows = periods == p
            self.coefs_[int(p)] = np.column_stack(
                [_quantreg(target[rows], matrix[rows], q) for q in QUANTILES]
            )
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Quantile forecasts in £/MWh; NaN where inputs are missing."""
        out = pd.DataFrame(np.nan, index=X.index, columns=list(QUANTILE_COLUMNS))
        ok = self.design.complete(X).to_numpy()
        if ok.any():
            matrix = _with_intercept(self.design.matrix(X[ok]))
            periods = X.loc[ok, "period_of_day"].to_numpy()
            z = np.full((len(matrix), len(QUANTILES)), np.nan)
            for p in np.unique(periods):
                rows = periods == p
                if int(p) in self.coefs_:
                    z[rows] = matrix[rows] @ self.coefs_[int(p)]
            out.loc[ok] = self.design.from_z(z)
        return sort_quantiles(out)


#: Days at the end of each training window held out to calibrate QRA.
QRA_CALIBRATION_DAYS = 182
#: How often LEAR is refitted while producing the calibration forecasts.
QRA_REFIT_DAYS = 14
#: Minimum days of history before the first calibration forecast.
QRA_MIN_ESTIMATION_DAYS = 90


class QRA:
    """Quantile regression averaging (Nowotarski and Weron, 2015).

    Combines several *point* forecasts into a quantile forecast. The point forecasts of LEAR,
    the seasonal naive, D-2 persistence and the weekday profile become the inputs of a
    quantile regression for each of q = 0.1, 0.5 and 0.9. The regression learns how much to
    trust each forecast and how wide the band should be. One regression is pooled across all
    half-hours, on £/MWh.

    The weights must be learnt from genuine *out-of-sample* forecasts, made the same way as
    the forecasts they will be applied to. So each refit replays a mini walk-forward over the
    last :data:`QRA_CALIBRATION_DAYS` of the training window: LEAR is refitted every
    :data:`QRA_REFIT_DAYS` days on data up to two days before each block (as in the main
    backtest) and forecasts it. The quantile regression is fitted on those forecasts, and LEAR
    is then refitted on the whole window to produce the inputs for the days being forecast.
    """

    name = "QRA"

    def _inputs(self, lear: LEAR, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        weekday = X[list(WEEKDAY_LAGS)].mean(axis=1, skipna=True)
        cols = [lear.predict_point(X), X["price_lag_7d"], X["price_lag_2d"], weekday]
        return np.column_stack([np.asarray(c, dtype=float) for c in cols])

    def _calibration_forecasts(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        days: np.ndarray,
        cal_start: dt.date,
    ) -> np.ndarray:
        """Out-of-sample inputs for every calibration row, from LEAR refitted block by block."""
        inputs = np.full((len(X), 4), np.nan)
        block_start = cal_start
        last = days.max()
        while block_start <= last:
            block_end = block_start + dt.timedelta(days=QRA_REFIT_DAYS)
            train = days <= block_start - dt.timedelta(days=2)
            rows = (days >= block_start) & (days < block_end)
            lear = LEAR().fit(X[train], y[train])
            inputs[rows] = self._inputs(lear, X[rows])
            block_start = block_end
        return inputs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QRA:  # noqa: N803
        """Replay LEAR over the calibration window, fit the QR, then refit LEAR on everything."""
        days = np.asarray(delivery_day(X.index))
        first, last = days.min(), days.max()
        cal_start = max(
            last - dt.timedelta(days=QRA_CALIBRATION_DAYS - 1),
            first + dt.timedelta(days=QRA_MIN_ESTIMATION_DAYS),
        )
        if cal_start > last:
            raise ValueError("QRA needs more history than this training window has")
        cal = days >= cal_start
        inputs = self._calibration_forecasts(X, y, days, cal_start)[cal]
        target = y[cal].to_numpy(dtype=float)
        ok = ~np.isnan(inputs).any(axis=1) & ~np.isnan(target)
        matrix = _with_intercept(inputs[ok])
        self.coefs_ = np.column_stack([_quantreg(target[ok], matrix, q) for q in QUANTILES])
        self.calibration_start_ = cal_start
        self.lear_ = LEAR().fit(X, y)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Quantile forecasts in £/MWh; NaN where any input forecast is missing."""
        inputs = self._inputs(self.lear_, X)
        values = _with_intercept(inputs) @ self.coefs_
        out = pd.DataFrame(values, index=X.index, columns=list(QUANTILE_COLUMNS))
        return sort_quantiles(out)
