import pytest

from gb_price_sandbox.metrics import mae, pinball_loss


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
