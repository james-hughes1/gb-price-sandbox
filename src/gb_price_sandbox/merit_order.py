"""A stylised GB merit order (supply stack) for explaining how wholesale prices are set.

Illustrative only. Capacities and costs are round numbers chosen to look like a recent GB
winter day, not a fitted model. Real bids also include start-up costs, scarcity, interconnector
flows, storage and strategic behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

THERM_MWH = 0.0293071  # 1 therm in MWh (thermal)
GAS_CO2_T_PER_MWH_TH = 0.184  # tCO2 per MWh of gas burned


@dataclass(frozen=True)
class Block:
    """One tranche of generation offered at a single marginal cost."""

    technology: str
    capacity_gw: float
    cost_gbp_mwh: float


def gas_plant_cost(efficiency: float, gas_p_therm: float, carbon_gbp_t: float, vom: float) -> float:
    """Short-run marginal cost (GBP/MWh electric) of a gas plant.

    Fuel and carbon per MWh of gas, divided by efficiency, plus variable O&M.
    """
    fuel = gas_p_therm / 100 / THERM_MWH
    carbon = GAS_CO2_T_PER_MWH_TH * carbon_gbp_t
    return (fuel + carbon) / efficiency + vom


def build_stack(wind_gw: float, gas_p_therm: float, carbon_gbp_t: float = 60.0) -> list[Block]:
    """The supply stack, cheapest first.

    Some wind is on older subsidies (Renewables Obligation, early CfDs) that pay out per MWh
    generated, so it will pay to keep running: it bids below zero. That is how GB gets
    negative prices.
    """
    blocks = [
        Block("Wind (subsidised)", 0.4 * wind_gw, -40.0),
        Block("Wind", 0.6 * wind_gw, 0.0),
        Block("Nuclear", 4.5, 5.0),
        Block("Biomass", 2.5, 60.0),
        Block("Gas CCGT (efficient)", 12.0, gas_plant_cost(0.55, gas_p_therm, carbon_gbp_t, 3.0)),
        Block("Gas CCGT (older)", 12.0, gas_plant_cost(0.48, gas_p_therm, carbon_gbp_t, 4.0)),
        Block("Gas peakers", 6.0, gas_plant_cost(0.35, gas_p_therm, carbon_gbp_t, 8.0)),
    ]
    return sorted((b for b in blocks if b.capacity_gw > 0), key=lambda b: b.cost_gbp_mwh)


def stack_frame(blocks: list[Block]) -> pd.DataFrame:
    """Stack as a table with cumulative capacity: ``start_gw``/``end_gw`` per block."""
    df = pd.DataFrame([b.__dict__ for b in blocks])
    df["end_gw"] = df["capacity_gw"].cumsum()
    df["start_gw"] = df["end_gw"] - df["capacity_gw"]
    return df


@dataclass(frozen=True)
class Clearing:
    """Result of clearing the stack against demand."""

    price_gbp_mwh: float | None  # None = demand exceeds the whole stack (scarcity)
    marginal_technology: str | None


def clear(blocks: list[Block], demand_gw: float) -> Clearing:
    """Price = cost of the most expensive block needed to meet demand (pay-as-clear)."""
    if demand_gw <= 0:
        raise ValueError("demand must be positive")
    supplied = 0.0
    for block in blocks:
        supplied += block.capacity_gw
        if supplied >= demand_gw - 1e-9:
            return Clearing(block.cost_gbp_mwh, block.technology)
    return Clearing(None, None)
