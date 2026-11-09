import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gb_price_sandbox.optimise import (
    BatterySpec,
    capture_rate,
    daily_dispatch,
    optimise_dispatch,
    summarise,
)
from gb_price_sandbox.timing import day_periods


def _series(values):
    return pd.Series(values, dtype=float)


# --- spec validation -------------------------------------------------------------------------


def test_default_spec_is_valid():
    BatterySpec()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"power_mw": 0},
        {"round_trip_efficiency": 1.5},
        {"soc_min": 0.6, "soc_initial": 0.5},
        {"soc_min": 0.5, "soc_initial": 0.5, "soc_max": 0.5},
        {"max_cycles_per_day": 0},
    ],
)
def test_invalid_spec_rejected(kwargs):
    with pytest.raises(ValueError):
        BatterySpec(**kwargs)


# --- hand-solved examples --------------------------------------------------------------------


def test_lossless_arbitrage_two_cycles():
    """1 MW / 0.5 MWh, lossless, starts empty. Each half-hour moves exactly 0.5 MWh.

    Prices 10, 50, 10, 50: charge, discharge, charge, discharge.
    Revenue = 2 * (50 - 10) * 0.5 = 40.
    """
    battery = BatterySpec(
        power_mw=1, energy_mwh=0.5, round_trip_efficiency=1, soc_initial=0, max_cycles_per_day=1e3
    )
    out = optimise_dispatch(_series([10, 50, 10, 50]), battery)
    assert out["revenue"].sum() == pytest.approx(40.0)
    np.testing.assert_allclose(out["charge_mw"], [1, 0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(out["discharge_mw"], [0, 1, 0, 1], atol=1e-9)
    np.testing.assert_allclose(out["soc_mwh"], [0.5, 0, 0.5, 0], atol=1e-9)


def test_efficiency_losses():
    """1 MW / 1 MWh, RTE 0.81 (0.9 each way), starts empty. Prices 0 then 100.

    Charge 1 MW for 0.5 h stores 0.45 MWh. Discharging d MW drains d * 0.5 / 0.9, so
    d = 0.81 MW empties it. Revenue = 100 * 0.81 * 0.5 = 40.5.
    """
    battery = BatterySpec(
        power_mw=1, energy_mwh=1, round_trip_efficiency=0.81, soc_initial=0, max_cycles_per_day=1e3
    )
    out = optimise_dispatch(_series([0, 100]), battery)
    assert out["revenue"].sum() == pytest.approx(40.5)
    assert out["discharge_mw"].iloc[1] == pytest.approx(0.81)


def test_negative_prices_binary_prevents_energy_burning():
    """1 MW / 1 MWh, RTE 0.81, starts half full (0.5 MWh headroom). Prices -50, -50.

    With the binary: the best you can do is fill the headroom. Storing 0.5 MWh takes
    0.5 / 0.9 MWh from the grid, so revenue = 50 * 0.5 / 0.9 = 27.78.

    Without it, the LP charges at 1 MW in both periods (+0.9 MWh stored) and discharges just
    enough at the same time (0.72 MW in total) to stay within the headroom.
    Net import = (2 - 0.72) * 0.5 MWh, so revenue = 50 * 0.64 = 32.0.
    """
    battery = BatterySpec(
        power_mw=1,
        energy_mwh=1,
        round_trip_efficiency=0.81,
        soc_initial=0.5,
        max_cycles_per_day=1e3,
    )
    prices = _series([-50, -50])

    milp = optimise_dispatch(prices, battery)
    assert milp["revenue"].sum() == pytest.approx(50 * 0.5 / 0.9)
    simultaneous = (milp["charge_mw"] > 1e-9) & (milp["discharge_mw"] > 1e-9)
    assert not simultaneous.any()

    lp = optimise_dispatch(prices, battery, relax_binary=True)
    assert lp["revenue"].sum() == pytest.approx(32.0)


def test_cycle_limit_binds():
    """Same as the lossless two-cycle case, but capped at one full cycle: revenue halves."""
    battery = BatterySpec(power_mw=1, energy_mwh=0.5, round_trip_efficiency=1, soc_initial=0)
    out = optimise_dispatch(_series([10, 50, 10, 50]), battery, max_cycles=1)
    assert out["revenue"].sum() == pytest.approx(20.0)
    assert summarise(out, battery)["cycles"] == pytest.approx(1.0)


def test_terminal_soc_respected():
    battery = BatterySpec(soc_initial=0.5)
    out = optimise_dispatch(_series([100, 100, 100, 100]), battery)
    # High flat prices tempt the battery to dump everything; it must end at least half full.
    assert out["soc_mwh"].iloc[-1] >= 0.5 * battery.energy_mwh - 1e-9
    assert out["revenue"].sum() == pytest.approx(0.0, abs=1e-6)


def test_nan_prices_rejected():
    with pytest.raises(ValueError):
        optimise_dispatch(_series([1.0, np.nan]), BatterySpec())


# --- daily dispatch and capture rate ---------------------------------------------------------


def _two_days():
    idx = day_periods(dt.date(2024, 6, 1)).append(day_periods(dt.date(2024, 6, 2)))
    hours = idx.tz_convert("Europe/London").hour
    # Cheap overnight, expensive evening peak.
    actual = pd.Series(np.where(hours < 6, 20.0, np.where(hours.isin([17, 18]), 120.0, 60.0)), idx)
    return actual


def test_perfect_foresight_beats_forecast():
    battery = BatterySpec()
    actual = _two_days()
    perfect = summarise(daily_dispatch(actual, None, battery), battery)

    # A forecast that has the peak in the morning instead of the evening.
    wrong = actual.copy()
    hours = actual.index.tz_convert("Europe/London").hour
    wrong[hours.isin([17, 18])] = 60.0
    wrong[hours.isin([7, 8])] = 120.0
    forecast = summarise(daily_dispatch(wrong, actual, battery), battery)

    assert perfect["revenue"] > forecast["revenue"]
    rate = capture_rate(forecast["revenue"], perfect["revenue"])
    assert rate < 1.0
    assert perfect["cycles"] <= 2 * battery.max_cycles_per_day + 1e-9


def test_daily_dispatch_is_per_day():
    battery = BatterySpec()
    out = daily_dispatch(_two_days(), None, battery)
    assert len(out) == 96
    # Each day ends at least back at the initial SoC.
    ends = out["soc_mwh"].groupby(out.index.tz_convert("Europe/London").date).last()
    assert (ends >= battery.soc_initial * battery.energy_mwh - 1e-9).all()


def test_capture_rate_requires_positive_benchmark():
    with pytest.raises(ValueError):
        capture_rate(1.0, 0.0)
