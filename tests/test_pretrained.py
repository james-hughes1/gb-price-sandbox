"""Pretrained Battery-page forecasts: saved once, then only read."""

import datetime as dt
import sys

import pytest

from gb_price_sandbox import app_data
from gb_price_sandbox.config import Settings
from gb_price_sandbox.optimise import PERFECT_FORESIGHT, BatterySpec
from gb_price_sandbox.optimise.battery import daily_dispatch
from gb_price_sandbox.pretrained import PRETRAINED_MODELS, load_pretrained, pretrain
from tests.test_backtest import synthetic_dataset

START, END = dt.date(2024, 3, 1), dt.date(2024, 3, 8)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setenv("GBPS_DATA_DIR", str(tmp_path))
    data = synthetic_dataset(dt.date(2024, 1, 1), dt.date(2024, 3, 8))
    pretrain(data, START, END, Settings(data_dir=tmp_path))
    return tmp_path


def test_pretrain_round_trips_through_disk(saved):
    loaded = load_pretrained(Settings(data_dir=saved))
    assert list(loaded.results) == list(PRETRAINED_MODELS)
    assert (loaded.test_start, loaded.test_end) == (START, END)
    for res in loaded.results.values():
        assert res["delivery_day"].min() == START
        assert isinstance(res["delivery_day"].iloc[0], dt.date)


def test_load_pretrained_none_when_missing(tmp_path):
    assert load_pretrained(Settings(data_dir=tmp_path)) is None


def test_battery_page_only_optimises(saved, monkeypatch):
    def no_training(*args, **kwargs):
        raise AssertionError("the Battery page must not train")

    # The module, not the same-named function the backtest package re-exports.
    module = sys.modules["gb_price_sandbox.backtest.walk_forward"]
    monkeypatch.setattr(module, "walk_forward", no_training)
    progress = []
    result = app_data.run_battery_page(
        1.0, 2.0, 0.9, 1.0, "Whole test period", lambda f, m: progress.append(f)
    )
    assert set(result.table.index) == {PERFECT_FORESIGHT, *PRETRAINED_MODELS}
    assert result.table.loc[PERFECT_FORESIGHT, "Capture rate"] == pytest.approx(1.0)
    assert progress == sorted(progress) and progress[-1] == pytest.approx(1.0)


def test_battery_page_explains_missing_forecasts(tmp_path, monkeypatch):
    monkeypatch.setenv("GBPS_DATA_DIR", str(tmp_path))
    with pytest.raises(app_data.PretrainedMissingError, match="pretrain"):
        app_data.run_battery_page(1.0, 2.0, 0.9, 1.0, "Last 3 months")


def test_daily_dispatch_reports_each_day():
    data = synthetic_dataset(dt.date(2024, 1, 1), dt.date(2024, 1, 4))
    calls = []
    daily_dispatch(data["price"], None, BatterySpec(), lambda d, t: calls.append((d, t)))
    assert calls == [(1, 3), (2, 3), (3, 3)]
