"""Tests for the time/cost estimate."""

import pytest

from app.estimate import Estimate, default_cost_cap, estimate_job, format_estimate


def test_estimate_five_hours_uses_provisional_constants() -> None:
    est = estimate_job(18000.0, 0.50)

    # low: 480 + 0.04*18000 + 0.30*18000 = 6600 s; high: 480 + 0.07*18000 + 0.55*18000 = 11640 s
    assert est.minutes_low == pytest.approx(110.0)
    assert est.minutes_high == pytest.approx(194.0)
    assert est.cost_low == pytest.approx(110.0 / 60 * 0.50)
    assert est.cost_high == pytest.approx(194.0 / 60 * 0.50)


def test_estimate_speed_factor_divides_everything() -> None:
    base = estimate_job(18000.0, 0.50)
    fast = estimate_job(18000.0, 0.50, speed_factor=2.0)

    assert fast.minutes_low == pytest.approx(base.minutes_low / 2)
    assert fast.minutes_high == pytest.approx(base.minutes_high / 2)
    assert fast.cost_high == pytest.approx(base.cost_high / 2)


def test_estimate_zero_duration_does_not_crash() -> None:
    est = estimate_job(0.0, 0.50)
    negative = estimate_job(-5.0, 0.50)

    assert est.minutes_low == pytest.approx(8.0)
    assert est.minutes_high == pytest.approx(8.0)
    assert est.cost_low == pytest.approx(8.0 / 60 * 0.50)
    assert negative == est


@pytest.mark.parametrize("speed", [0.0, -1.0])
def test_estimate_rejects_non_positive_speed_factor(speed: float) -> None:
    with pytest.raises(ValueError, match="speed_factor"):
        estimate_job(100.0, 0.50, speed_factor=speed)


def test_default_cost_cap_has_one_dollar_floor() -> None:
    assert default_cost_cap(Estimate(1, 2, 0.10, 0.20)) == 1.0
    assert default_cost_cap(Estimate(1, 2, 0.10, 0.50)) == 1.0
    assert default_cost_cap(Estimate(1, 2, 0.10, 1.60)) == pytest.approx(3.2)


def test_format_estimate_hours_and_cost() -> None:
    est = Estimate(minutes_low=110.0, minutes_high=180.0, cost_low=0.9, cost_high=1.6)

    assert format_estimate(est) == "Est. 1h 50m to 3h, about $0.90 to $1.60"


def test_format_estimate_rounds_to_five_minutes_and_cents() -> None:
    est = Estimate(minutes_low=47.4, minutes_high=194.0, cost_low=0.9167, cost_high=1.6167)

    assert format_estimate(est) == "Est. 45m to 3h 15m, about $0.92 to $1.62"


def test_format_estimate_single_value_when_low_and_high_match() -> None:
    est = Estimate(minutes_low=8.0, minutes_high=8.0, cost_low=0.0667, cost_high=0.0667)

    assert format_estimate(est) == "Est. 10m, about $0.07"
