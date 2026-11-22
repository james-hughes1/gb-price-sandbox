"""Turn backtest forecasts into battery revenue and capture rates."""

from __future__ import annotations

from collections.abc import Callable

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
    results: dict[str, pd.DataFrame],
    battery: BatterySpec,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Dispatch on each model's median forecast, settle at realised prices, compare to perfect.

    Only days where every model and the realised price are complete are used, so all models
    are scored on the same days.

    Args:
        results: Backtest output per model (from ``walk_forward``), sharing an index.
        battery: Battery parameters.
        progress: Called as ``progress(fraction_done, message)`` as days are solved.

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

    runs = [(PERFECT_FORESIGHT, actual, None)] + [
        (name, res.loc[keep, "q50"], actual) for name, res in results.items()
    ]
    schedules = {}
    for i, (name, decision, settlement) in enumerate(runs):

        def report(done: int, total: int, i: int = i, name: str = name) -> None:
            if progress is not None:
                progress((i + done / total) / len(runs), f"{name}: day {done} of {total}")

        schedules[name] = daily_dispatch(decision, settlement, battery, report)

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
