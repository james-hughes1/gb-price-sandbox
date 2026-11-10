"""Leaderboard and error breakdowns for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gb_price_sandbox.metrics import crps_from_quantiles, interval_coverage, mae, pinball_loss
from gb_price_sandbox.models.base import QUANTILE_COLUMNS, QUANTILES


def common_rows(results: dict[str, pd.DataFrame]) -> pd.Index:
    """Periods where the target and every model's quantiles are all present.

    Models are compared on exactly the same periods, so none benefits from skipping hard
    ones.
    """
    index = None
    for res in results.values():
        ok = res.index[res[[*QUANTILE_COLUMNS, "y"]].notna().all(axis=1)]
        index = ok if index is None else index.intersection(ok)
    return index if index is not None else pd.Index([])


def leaderboard(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Score each model on the common periods.

    Columns: ``MAE`` (of the median), ``Pinball`` (mean over the quantiles), ``CRPS``
    (approximate, from the quantiles), ``Coverage 80%`` (share of outcomes inside the q10-q90
    band; a calibrated model scores about 0.8, a point forecast about 0), ``Periods``.
    Sorted by CRPS, best first.
    """
    rows = common_rows(results)
    table = {}
    for name, res in results.items():
        r = res.loc[rows]
        y = r["y"].to_numpy()
        q = r[list(QUANTILE_COLUMNS)].to_numpy()
        table[name] = {
            "MAE": mae(y, r["q50"]),
            "Pinball": float(
                np.mean([pinball_loss(y, q[:, k], tau) for k, tau in enumerate(QUANTILES)])
            ),
            "CRPS": crps_from_quantiles(y, q, QUANTILES),
            "Coverage 80%": interval_coverage(y, r["q10"], r["q90"]),
            "Periods": len(r),
        }
    return pd.DataFrame.from_dict(table, orient="index").sort_values("CRPS")


def error_by_period_of_day(
    results: dict[str, pd.DataFrame], period_of_day: pd.Series
) -> pd.DataFrame:
    """MAE of the median forecast by local half-hour of day (rows) for each model (columns)."""
    rows = common_rows(results)
    slot = period_of_day.loc[rows]
    out = {}
    for name, res in results.items():
        err = (res.loc[rows, "q50"] - res.loc[rows, "y"]).abs()
        out[name] = err.groupby(slot).mean()
    return pd.DataFrame(out)
