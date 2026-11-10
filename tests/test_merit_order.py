import pytest

from gb_price_sandbox.merit_order import (
    Block,
    build_stack,
    clear,
    gas_plant_cost,
    stack_frame,
)


def test_gas_cost_hand_calc():
    # 100 p/therm = 1 GBP / 0.0293071 MWh = 34.12 GBP/MWh_th; no carbon; 50% efficient.
    assert gas_plant_cost(0.5, 100, 0, 0) == pytest.approx(2 * 34.1213, rel=1e-4)


def test_pay_as_clear():
    blocks = [Block("a", 10, 0), Block("b", 10, 50), Block("c", 10, 100)]
    assert clear(blocks, 5).price_gbp_mwh == 0
    assert clear(blocks, 10).price_gbp_mwh == 0  # exactly at a boundary: cheaper block suffices
    assert clear(blocks, 15) == clear(blocks, 20)
    assert clear(blocks, 25).marginal_technology == "c"
    assert clear(blocks, 31).price_gbp_mwh is None  # scarcity


def test_more_wind_never_raises_price():
    prices = [clear(build_stack(w, 80), 30).price_gbp_mwh for w in range(0, 25, 2)]
    assert all(a >= b for a, b in zip(prices, prices[1:], strict=False))


def test_high_wind_low_demand_goes_negative():
    assert clear(build_stack(20, 80), 6).price_gbp_mwh < 0


def test_gas_price_sets_price_when_gas_is_marginal():
    lo = clear(build_stack(5, 50), 30)
    hi = clear(build_stack(5, 150), 30)
    assert lo.marginal_technology.startswith("Gas") and hi.price_gbp_mwh > lo.price_gbp_mwh


def test_stack_frame_is_contiguous():
    df = stack_frame(build_stack(10, 80))
    assert df["start_gw"].iloc[0] == 0
    assert (df["start_gw"].iloc[1:].to_numpy() == df["end_gw"].iloc[:-1].to_numpy()).all()
    assert df["cost_gbp_mwh"].is_monotonic_increasing
