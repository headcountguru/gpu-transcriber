"""Time and cost estimate shown to the user before a job starts."""

from dataclasses import dataclass

# Provisional T4-baseline constants. DELIBERATELY UNCHANGED after the first live run: only a
# 3-minute sample was measured (about 11 s of GPU work for 180 s of audio on an RTX A5000), which
# ran far faster than these figures, so they stay conservative for long recordings. The real
# speed on a 5-hour recording is unmeasured; revisit once a long run has been timed.
FIXED_OVERHEAD_S = 480.0
TRANSCRIBE_FACTOR_LOW = 0.04
TRANSCRIBE_FACTOR_HIGH = 0.07
DIARIZE_FACTOR_LOW = 0.30
DIARIZE_FACTOR_HIGH = 0.55

MIN_COST_CAP = 1.0
COST_CAP_MULTIPLE = 2.0

_SECONDS_PER_MINUTE = 60.0
_MINUTES_PER_HOUR = 60
_ROUND_TO_MINUTES = 5


@dataclass(frozen=True)
class Estimate:
    """Low and high bounds for run time (minutes) and rental cost (dollars)."""

    minutes_low: float
    minutes_high: float
    cost_low: float
    cost_high: float


def estimate_job(duration_s: float, price_per_hour: float, speed_factor: float = 1.0) -> Estimate:
    """Estimate time and cost for ``duration_s`` of audio at ``price_per_hour`` rental.

    ``speed_factor`` is how much faster than the T4 baseline the machine is. A non-positive
    ``duration_s`` yields the fixed-overhead-only estimate.
    """
    if speed_factor <= 0:
        raise ValueError(f"speed_factor must be positive, got {speed_factor}")
    audio_s = max(duration_s, 0.0)
    low_s = (
        FIXED_OVERHEAD_S + (TRANSCRIBE_FACTOR_LOW + DIARIZE_FACTOR_LOW) * audio_s
    ) / speed_factor
    high_s = (
        FIXED_OVERHEAD_S + (TRANSCRIBE_FACTOR_HIGH + DIARIZE_FACTOR_HIGH) * audio_s
    ) / speed_factor
    minutes_low = low_s / _SECONDS_PER_MINUTE
    minutes_high = high_s / _SECONDS_PER_MINUTE
    return Estimate(
        minutes_low=minutes_low,
        minutes_high=minutes_high,
        cost_low=minutes_low / _MINUTES_PER_HOUR * price_per_hour,
        cost_high=minutes_high / _MINUTES_PER_HOUR * price_per_hour,
    )


def default_cost_cap(estimate: Estimate) -> float:
    """Default spending cap: twice the high estimate, but never below one dollar."""
    return max(MIN_COST_CAP, COST_CAP_MULTIPLE * estimate.cost_high)


def _format_minutes(minutes: float) -> str:
    rounded = int(minutes / _ROUND_TO_MINUTES + 0.5) * _ROUND_TO_MINUTES
    hours, mins = divmod(rounded, _MINUTES_PER_HOUR)
    if hours == 0:
        return f"{mins}m"
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"


def _format_cost(cost: float) -> str:
    return f"${cost:.2f}"


def format_estimate(estimate: Estimate) -> str:
    """Render e.g. ``Est. 1h 50m to 3h, about $0.90 to $1.60``."""
    time_low = _format_minutes(estimate.minutes_low)
    time_high = _format_minutes(estimate.minutes_high)
    cost_low = _format_cost(estimate.cost_low)
    cost_high = _format_cost(estimate.cost_high)
    time_text = time_low if time_low == time_high else f"{time_low} to {time_high}"
    cost_text = cost_low if cost_low == cost_high else f"{cost_low} to {cost_high}"
    return f"Est. {time_text}, about {cost_text}"
