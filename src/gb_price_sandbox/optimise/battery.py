"""Battery specification and dispatch optimisation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BatterySpec:
    """Physical and commercial parameters of a battery.

    Attributes:
        power_mw: Maximum charge/discharge power.
        energy_mwh: Usable energy capacity.
        round_trip_efficiency: Fraction of energy returned over a full cycle, in (0, 1].
        soc_min: Minimum state of charge as a fraction of ``energy_mwh``.
        soc_max: Maximum state of charge as a fraction of ``energy_mwh``.
        soc_initial: Initial state of charge as a fraction of ``energy_mwh``.
        max_cycles_per_day: Limit on full-equivalent discharge cycles per day.
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
        if self.max_cycles_per_day <= 0:
            raise ValueError("max_cycles_per_day must be positive")


def optimise_dispatch(prices: pd.Series, battery: BatterySpec) -> pd.DataFrame:
    """Solve the revenue-maximising dispatch for a half-hourly price series (MILP).

    Uses a binary per period to forbid simultaneous charging and discharging, which
    otherwise becomes profitable at negative prices.

    Args:
        prices: Half-hourly prices (GBP/MWh) indexed by period start.
        battery: Battery parameters.

    Returns:
        DataFrame indexed like ``prices`` with ``charge_mw``, ``discharge_mw`` and ``soc_mwh``.
    """
    raise NotImplementedError
