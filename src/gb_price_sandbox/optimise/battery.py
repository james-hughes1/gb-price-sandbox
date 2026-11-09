"""Battery specification and revenue-maximising dispatch (MILP via PuLP + HiGHS).

Sign conventions: ``charge_mw`` is power drawn from the grid and ``discharge_mw`` is power
exported to the grid, both non-negative and measured at the grid connection. Revenue in a
period is ``price * (discharge - charge) * period_hours``. Round-trip losses are split
evenly between charging and discharging (each leg has efficiency ``sqrt(rte)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import pulp

from gb_price_sandbox.timing import PERIOD_HOURS, delivery_day


@dataclass(frozen=True)
class BatterySpec:
    """Physical and commercial parameters of a battery.

    Attributes:
        power_mw: Maximum charge/discharge power.
        energy_mwh: Nameplate energy capacity.
        round_trip_efficiency: Fraction of energy returned over a full cycle, in (0, 1].
        soc_min: Minimum state of charge as a fraction of ``energy_mwh``.
        soc_max: Maximum state of charge as a fraction of ``energy_mwh``.
        soc_initial: State of charge at the start of each optimisation window, as a
            fraction of ``energy_mwh``. The window must end at least this full.
        max_cycles_per_day: Limit on full-equivalent cycles per day, where one cycle is
            discharging the usable energy ``(soc_max - soc_min) * energy_mwh``.
    """

    power_mw: float = 1.0
    energy_mwh: float = 2.0
    round_trip_efficiency: float = 0.9
    soc_min: float = 0.0
    soc_max: float = 1.0
    soc_initial: float = 0.5
    max_cycles_per_day: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.power_mw <= 0 or self.energy_mwh <= 0:
            raise ValueError("power_mw and energy_mwh must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("round_trip_efficiency must be in (0, 1]")
        if not 0 <= self.soc_min <= self.soc_initial <= self.soc_max <= 1:
            raise ValueError("require 0 <= soc_min <= soc_initial <= soc_max <= 1")
        if self.soc_max == self.soc_min:
            raise ValueError("soc_max must exceed soc_min")
        if self.max_cycles_per_day <= 0:
            raise ValueError("max_cycles_per_day must be positive")

    @property
    def usable_energy_mwh(self) -> float:
        """Energy between the SoC bounds."""
        return (self.soc_max - self.soc_min) * self.energy_mwh

    @property
    def leg_efficiency(self) -> float:
        """One-way (charge or discharge) efficiency."""
        return math.sqrt(self.round_trip_efficiency)


def optimise_dispatch(
    prices: pd.Series,
    battery: BatterySpec,
    *,
    period_hours: float = PERIOD_HOURS,
    max_cycles: float | None = None,
    relax_binary: bool = False,
) -> pd.DataFrame:
    """Solve the revenue-maximising dispatch for a price series with perfect knowledge of it.

    A binary per period forbids simultaneous charging and discharging. Without it, at negative
    prices the optimiser would charge and discharge at once to burn energy through
    round-trip losses and get paid for consuming it, which a real battery cannot do.

    Args:
        prices: Prices (GBP/MWh), one per period, in time order.
        battery: Battery parameters.
        period_hours: Length of each period in hours.
        max_cycles: Cycle limit for the whole window. Defaults to
            ``battery.max_cycles_per_day`` pro-rated to the window length.
        relax_binary: Drop the no-simultaneous-charge/discharge binary, which turns the
            MILP into an LP. Only useful for showing why the binary matters.

    Returns:
        DataFrame indexed like ``prices`` with ``price``, ``charge_mw``, ``discharge_mw``,
        ``soc_mwh`` (at the end of each period) and ``revenue`` (GBP).
    """
    if prices.isna().any():
        raise ValueError("prices contain NaN; fill or drop them before optimising")
    n = len(prices)
    if n == 0:
        raise ValueError("prices is empty")
    if max_cycles is None:
        max_cycles = battery.max_cycles_per_day * n * period_hours / 24.0

    eta = battery.leg_efficiency
    e_cap = battery.energy_mwh
    p_max = battery.power_mw
    t_range = range(n)
    price = prices.to_numpy(dtype=float)

    prob = pulp.LpProblem("battery_dispatch", pulp.LpMaximize)
    charge = [prob.add_variable(f"c_{t}", lowBound=0, upBound=p_max) for t in t_range]
    discharge = [prob.add_variable(f"d_{t}", lowBound=0, upBound=p_max) for t in t_range]
    soc_lo, soc_hi = battery.soc_min * e_cap, battery.soc_max * e_cap
    soc = [prob.add_variable(f"soc_{t}", lowBound=soc_lo, upBound=soc_hi) for t in t_range]

    prob += pulp.lpSum(price[t] * (discharge[t] - charge[t]) * period_hours for t in t_range)

    soc_prev: pulp.LpAffineExpression | float = battery.soc_initial * e_cap
    for t in t_range:
        prob += (
            soc[t] == soc_prev + (eta * charge[t] - discharge[t] / eta) * period_hours,
            f"soc_balance_{t}",
        )
        soc_prev = soc[t]
        if not relax_binary:
            is_charging = prob.add_variable(f"u_{t}", cat=pulp.LpBinary)
            prob += charge[t] <= p_max * is_charging, f"charge_mode_{t}"
            prob += discharge[t] <= p_max * (1 - is_charging), f"discharge_mode_{t}"

    prob += soc[-1] >= battery.soc_initial * e_cap, "terminal_soc"
    prob += (
        pulp.lpSum(discharge) * period_hours <= max_cycles * battery.usable_energy_mwh,
        "cycle_limit",
    )

    stats = prob.solve(pulp.HiGHS(msg=False))
    if stats.status.name != "Optimal":
        raise RuntimeError(f"dispatch optimisation failed: {stats.status.name}")

    out = pd.DataFrame(
        {
            "price": price,
            "charge_mw": [v.value() for v in charge],
            "discharge_mw": [v.value() for v in discharge],
            "soc_mwh": [v.value() for v in soc],
        },
        index=prices.index,
    )
    out["revenue"] = settle(out, prices, period_hours=period_hours)
    return out


def settle(
    schedule: pd.DataFrame, prices: pd.Series, *, period_hours: float = PERIOD_HOURS
) -> pd.Series:
    """Per-period revenue (GBP) of a dispatch schedule settled at ``prices``."""
    net_mw = schedule["discharge_mw"] - schedule["charge_mw"]
    return prices * net_mw * period_hours


def daily_dispatch(
    decision_prices: pd.Series,
    settlement_prices: pd.Series | None,
    battery: BatterySpec,
) -> pd.DataFrame:
    """Dispatch one GB delivery day at a time, settling each schedule at realised prices.

    Mirrors trading day-ahead: each day's schedule is fixed from ``decision_prices`` (a
    forecast, or the actuals for perfect foresight) and then paid ``settlement_prices``.
    Every day starts at ``battery.soc_initial`` and must end at least that full, so days
    are independent.

    Args:
        decision_prices: Half-hourly prices the schedule is optimised against, indexed by
            tz-aware UTC period start.
        settlement_prices: Realised half-hourly prices, same index. ``None`` settles at
            ``decision_prices`` (perfect foresight).
        battery: Battery parameters.

    Returns:
        Per-period schedule as from :func:`optimise_dispatch`, plus ``settlement_price``,
        with ``revenue`` computed at settlement prices.
    """
    if settlement_prices is None:
        settlement_prices = decision_prices
    if not decision_prices.index.equals(settlement_prices.index):
        raise ValueError("decision and settlement prices must share an index")

    days = []
    for _, day_prices in decision_prices.groupby(delivery_day(decision_prices.index)):
        days.append(optimise_dispatch(day_prices, battery))
    schedule = pd.concat(days)
    schedule["settlement_price"] = settlement_prices
    schedule["revenue"] = settle(schedule, settlement_prices)
    return schedule


def summarise(schedule: pd.DataFrame, battery: BatterySpec) -> dict[str, float]:
    """Total revenue (GBP) and full-equivalent cycles of a dispatch schedule."""
    discharged_mwh = float(schedule["discharge_mw"].sum() * PERIOD_HOURS)
    return {
        "revenue": float(schedule["revenue"].sum()),
        "cycles": discharged_mwh / battery.usable_energy_mwh,
    }


def capture_rate(forecast_revenue: float, perfect_revenue: float) -> float:
    """Share of perfect-foresight revenue captured by forecast-driven dispatch.

    Can be negative if forecast errors make the battery lose money.
    """
    if perfect_revenue <= 0:
        raise ValueError("perfect-foresight revenue must be positive to define a capture rate")
    return forecast_revenue / perfect_revenue
