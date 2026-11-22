"""Leaderboard and error breakdowns for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gb_price_sandbox.metrics import crps_from_quantiles, interval_coverage, mae, pinball_loss, rmse
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


#: Metrics the leaderboard can rank by: name -> (description, lower is better).
METRICS: dict[str, str] = {
    "MAE": "Mean absolute error of the median (or mean) forecast, £/MWh.",
    "RMSE": "Root mean squared error: punishes large misses, i.e. price spikes, much more.",
    "Pinball": "Average pinball loss over q10, q50 and q90: the standard quantile score.",
    "CRPS": "Scores the whole forecast distribution; equals MAE for a point forecast.",
    "Coverage error": "Distance of the 10-90% band's coverage from the ideal 80%.",
}


def leaderboard(results: dict[str, pd.DataFrame], sort_by: str = "CRPS") -> pd.DataFrame:
    """Score each model on the common periods.

    Columns: ``MAE`` and ``RMSE`` (of the median), ``Pinball`` (mean over the quantiles),
    ``CRPS`` (approximate, from the quantiles), ``Coverage 80%`` (share of outcomes inside the
    q10-q90 band; a calibrated model scores about 0.8, a point forecast about 0),
    ``Coverage error`` (``|Coverage 80% - 0.8|``) and ``Periods``. Sorted by ``sort_by``,
    best (lowest) first.
    """
    rows = common_rows(results)
    table = {}
    for name, res in results.items():
        r = res.loc[rows]
        y = r["y"].to_numpy()
        q = r[list(QUANTILE_COLUMNS)].to_numpy()
        coverage = interval_coverage(y, r["q10"], r["q90"])
        table[name] = {
            "MAE": mae(y, r["q50"]),
            "RMSE": rmse(y, r["q50"]),
            "Pinball": float(
                np.mean([pinball_loss(y, q[:, k], tau) for k, tau in enumerate(QUANTILES)])
            ),
            "CRPS": crps_from_quantiles(y, q, QUANTILES),
            "Coverage 80%": coverage,
            "Coverage error": abs(coverage - 0.8),
            "Periods": len(r),
        }
    return pd.DataFrame.from_dict(table, orient="index").sort_values(sort_by)
