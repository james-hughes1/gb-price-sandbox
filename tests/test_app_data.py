import datetime as dt

import pytest

from gb_price_sandbox import app_data


@pytest.fixture
def fixed_today(monkeypatch):
    monkeypatch.setattr(app_data, "today_london", lambda: dt.date(2026, 11, 9))


def test_default_test_end_is_month_start(fixed_today):
    assert app_data.default_test_end() == dt.date(2026, 11, 1)


@pytest.mark.parametrize(
    ("label", "start"),
    [
        ("Last 3 months", dt.date(2026, 8, 1)),
        ("Last 12 months", dt.date(2025, 11, 1)),
        ("Whole test period", app_data.DEFAULT_TEST_START),
    ],
)
def test_battery_window(fixed_today, label, start):
    assert app_data.battery_window(label) == (start, dt.date(2026, 11, 1))


def test_model_colors_fixed_per_entity():
    colors = app_data.model_colors(dark=False)
    assert colors["Actual"] == colors["Perfect foresight"]  # both in neutral ink
    assert colors["Seasonal naive"] != colors["LEAR"]
    assert app_data.model_colors(dark=True)["LEAR"] != colors["LEAR"]  # own dark step
