import pytest

from gb_price_sandbox.optimise import BatterySpec


def test_default_spec_is_valid():
    BatterySpec()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"power_mw": 0},
        {"round_trip_efficiency": 1.5},
        {"soc_min": 0.6, "soc_initial": 0.5},
        {"max_cycles_per_day": 0},
    ],
)
def test_invalid_spec_rejected(kwargs):
    with pytest.raises(ValueError):
        BatterySpec(**kwargs)
