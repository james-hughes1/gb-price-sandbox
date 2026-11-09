import numpy as np
import pytest
from scipy import stats

from gb_price_sandbox.metrics import crps_from_quantiles, interval_coverage, mae, pinball_loss


def test_mae():
    assert mae([1, 2, 3], [1, 3, 5]) == pytest.approx(1.0)


def test_pinball_median_is_half_mae():
    y, p = [1.0, 2.0, 3.0], [2.0, 2.0, 2.0]
    assert pinball_loss(y, p, 0.5) == pytest.approx(0.5 * mae(y, p))


def test_pinball_asymmetry():
    # Under-forecast at q=0.9 costs 0.9 per unit; over-forecast costs 0.1.
    assert pinball_loss([10.0], [0.0], 0.9) == pytest.approx(9.0)
    assert pinball_loss([0.0], [10.0], 0.9) == pytest.approx(1.0)


def test_pinball_rejects_bad_q():
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 1.0)


def test_interval_coverage_inclusive_bounds():
    y = [0.0, 1.0, 2.0, 3.0]
    assert interval_coverage(y, [1.0] * 4, [2.0] * 4) == pytest.approx(0.5)


def test_crps_point_forecast_equals_mae():
    # A degenerate forecast (all quantiles equal) has CRPS == absolute error, for any grid.
    y = np.array([1.0, -2.0, 5.0])
    point = np.array([0.0, 0.0, 0.0])
    qs = [i / 100 for i in range(1, 100)]
    q_pred = np.repeat(point[:, None], len(qs), axis=1)
    # The midpoint-rule sum of 2*pinball over a uniform grid integrates to |y - x| exactly.
    assert crps_from_quantiles(y, q_pred, qs) == pytest.approx(mae(y, point), rel=1e-12)


def test_crps_matches_gaussian_closed_form():
    # CRPS(N(0, 1), y) = y(2Phi(y) - 1) + 2phi(y) - 1/sqrt(pi).
    y = np.array([-1.5, 0.0, 0.7, 2.0])
    k = 999
    qs = [(i + 0.5) / k for i in range(k)]
    q_pred = np.tile(stats.norm.ppf(qs), (len(y), 1))
    exact = y * (2 * stats.norm.cdf(y) - 1) + 2 * stats.norm.pdf(y) - 1 / np.sqrt(np.pi)
    assert crps_from_quantiles(y, q_pred, qs) == pytest.approx(exact.mean(), rel=1e-3)


def test_crps_shape_check():
    with pytest.raises(ValueError):
        crps_from_quantiles([1.0], [[1.0, 2.0]], [0.1, 0.5, 0.9])
