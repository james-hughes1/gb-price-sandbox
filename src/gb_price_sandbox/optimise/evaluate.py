"""Turn backtest forecasts into battery revenue and capture rates."""

from __future__ import annotations

import pandas as pd

from gb_price_sandbox.optimise.battery import BatterySpec, capture_rate, daily_dispatch, summarise

PERFECT_FORESIGHT = "Perfect foresight"


def dispatchable_days(results: dict[str, pd.DataFrame]) -> list:
    """Delivery days on which the realised price and every model's median are all complete."""
    days = None
    for res in results.values():
        ok = res[["q50", "y"]].notna().all(axis=1).groupby(res["delivery_day"]).all()
        good = set(ok[ok].index)
        days = good if days is None else days & good
    return sorted(days or [])


def battery_backtest(
    results: dict[str, pd.DataFrame], battery: BatterySpec
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Dispatch on each model's median forecast, settle at realised prices, compare to perfect.

    Only days where every model and the realised price are complete are used, so all models
    are scored on the same days.

    Args:
        results: Backtest output per model (from ``walk_forward``), sharing an index.
        battery: Battery parameters.

    Returns:
        A summary table (one row per model plus perfect foresight) with ``Revenue (GBP)``,
        ``Cycles``, ``Capture rate`` and ``Days``, and the per-period schedules by name.
    """
    days = dispatchable_days(results)
    if not days:
        raise ValueError("no day has complete prices and forecasts for every model")
    first = next(iter(results.values()))
    keep = first["delivery_day"].isin(days)
    actual = first.loc[keep, "y"]

    schedules = {PERFECT_FORESIGHT: daily_dispatch(actual, None, battery)}
    for name, res in results.items():
        schedules[name] = daily_dispatch(res.loc[keep, "q50"], actual, battery)

    perfect = summarise(schedules[PERFECT_FORESIGHT], battery)["revenue"]
    rows = {}
    for name, sched in schedules.items():
        s = summarise(sched, battery)
        rows[name] = {
            "Revenue (GBP)": s["revenue"],
            "Cycles": s["cycles"],
            "Capture rate": capture_rate(s["revenue"], perfect),
            "Days": len(days),
        }
    return pd.DataFrame.from_dict(rows, orient="index"), schedules
