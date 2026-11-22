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


def _exploration_frame():
    from tests.test_explore import _build, _dataset

    return _build(_dataset())


def test_filter_feature_frame_by_year_and_time_of_day():
    frame = _exploration_frame()
    out = app_data.filter_feature_frame(frame, (2024, 2024), (20, 29))
    assert out["period_of_day"].between(20, 29).all()
    assert len(out) == (frame["period_of_day"].between(20, 29)).sum()
    assert app_data.filter_feature_frame(frame, (2030, 2030), (0, 47)).empty


def test_feature_view_for_every_feature():
    frame = _exploration_frame()
    for key, label in app_data.feature_selector():
        view = app_data.feature_view(frame, key)
        assert view.spec.key == key and label.endswith(view.spec.title)
        for v in view.variants:
            if v.kind == "continuous":
                assert v.column in view.histograms
            else:
                assert v.column in view.categories
        for spearman, pearson in view.correlations.values():
            assert -1 <= spearman <= 1 and -1 <= pearson <= 1


def test_feature_correlations_labels_match_matrix():
    frame = _exploration_frame()
    matrix, labels, owner = app_data.feature_correlations(frame, "pearson", True)
    assert matrix.shape == (len(labels), len(labels))
    assert set(owner) == set(labels)
    assert owner["Price"] == "price"
