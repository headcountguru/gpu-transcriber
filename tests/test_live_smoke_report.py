"""Tests for the pure parts of the live smoke script (report, timing, spend cap, status parsing)."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from app.vast_client import Offer, VastError
from common.protocol import RemoteStatus
from scripts.live_smoke import (
    Assumption,
    EndpointTracker,
    PollResult,
    build_report,
    format_report,
    judge_reliability_filter,
    parse_ffmpeg_duration,
    parse_status,
    poll_job,
    spend_cap_reached,
    stage_durations,
)

ROOT = Path(__file__).resolve().parent.parent


def _offer() -> Offer:
    return Offer(
        id=7,
        gpu_name="RTX 4090",
        gpu_ram_gb=24.0,
        dph_total=0.40,
        inet_down_mbps=900.0,
        reliability=0.99,
        datacenter=True,
        cuda_max_good=12.6,
    )


def _timings() -> dict[str, float]:
    return {
        "startup": 120.0,
        "upload": 60.0,
        "preparing": 30.0,
        "transcribing": 700.0,
        "labeling": 250.0,
        "aligning": 60.0,
        "writing": 10.0,
        "download": 30.0,
    }


def test_format_report_includes_ratio_and_cost() -> None:
    # Arrange: 1260 s of work for 3600 s of audio is exactly 0.35x
    report = format_report(_timings(), _offer(), audio_seconds=3600.0, total_cost=0.14)

    # Assert
    assert "RTX 4090" in report
    assert "$0.14" in report
    assert "$0.40" in report
    assert "0.35x" in report
    for label in ("startup", "upload", "transcribing", "labeling", "download"):
        assert label in report


def test_format_report_handles_zero_audio_length() -> None:
    report = format_report(_timings(), _offer(), audio_seconds=0.0, total_cost=0.0)

    assert "n/a" in report


def test_stage_durations_use_gap_to_next_stage_and_end() -> None:
    first_seen = {"preparing": 10.0, "transcribing": 40.0, "labeling": 100.0}

    durations = stage_durations(first_seen, end=130.0)

    assert durations == {"preparing": 30.0, "transcribing": 60.0, "labeling": 30.0}


def test_stage_durations_empty_when_no_stage_seen() -> None:
    assert stage_durations({}, end=50.0) == {}


def test_spend_cap_stops_at_the_cap() -> None:
    assert not spend_cap_reached(hours=0.5, price_per_hour=1.0, max_spend=1.0)
    assert spend_cap_reached(hours=1.0, price_per_hour=1.0, max_spend=1.0)
    assert spend_cap_reached(hours=2.0, price_per_hour=1.0, max_spend=1.0)


def test_parse_status_is_defensive() -> None:
    good = '{"state": "running", "stage": "transcribing", "percent": 5, "message": "x"}'

    status = parse_status(good)

    assert status is not None and status.stage == "transcribing"
    assert parse_status(None) is None
    assert parse_status("") is None
    assert parse_status("{not json") is None
    assert parse_status('{"unexpected": 1}') is None
    assert parse_status("[1, 2]") is None


def test_report_never_contains_secrets() -> None:
    vast_key = "vast-key-AAAA1111"
    hf_token = "hf_BBBB2222"
    assumptions = [
        Assumption("SSH works", "PASS", f"connected; key={vast_key}"),
        Assumption("Search", "FAIL", f"HTTP 500 with token {hf_token}"),
    ]

    report = build_report(
        timings=_timings(),
        offer=_offer(),
        audio_seconds=3600.0,
        total_cost=0.14,
        command="python scripts/live_smoke.py --audio a.wav --max-spend 1.0",
        assumptions=assumptions,
        notes=[f"stopped early {vast_key}"],
        secrets=[vast_key, hf_token],
    )

    assert vast_key not in report
    assert hf_token not in report
    assert "PASS" in report and "FAIL" in report
    assert "python scripts/live_smoke.py" in report


def test_help_works_without_env_keys() -> None:
    env = dict(os.environ)
    env.pop("VAST_API_KEY", None)
    env.pop("HF_TOKEN", None)

    result = subprocess.run(
        [sys.executable, "scripts/live_smoke.py", "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "--audio" in result.stdout and "--max-spend" in result.stdout


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeSession:
    """Returns the queued status texts one per read, then repeats the last."""

    def __init__(self, texts: list[str | None]) -> None:
        self._texts = texts
        self._reads = 0

    def read_text(self, remote_path: str) -> str | None:
        text = self._texts[min(self._reads, len(self._texts) - 1)]
        self._reads += 1
        return text


def _status(state: str, stage: str, percent: int = 0) -> str:
    return RemoteStatus(state, stage, percent, "m").to_json()


def _poll(session: _FakeSession, clock: _Clock, max_spend: float = 100.0) -> PollResult:
    return poll_job(
        session,  # type: ignore[arg-type]
        "/work/job/status.json",
        billed_hours=lambda: clock.now / 3600.0,
        price_per_hour=0.4,
        max_spend=max_spend,
        timeout_s=10_000.0,
        clock=clock,
        sleep=clock.sleep,
    )


def test_poll_job_records_first_sighting_of_each_stage_and_skips_bad_files() -> None:
    clock = _Clock()
    session = _FakeSession(
        [
            None,
            "{corrupt",
            _status("running", "preparing"),
            _status("running", "preparing"),
            _status("running", "transcribing", 10),
            _status("done", "done", 100),
        ]
    )

    result = _poll(session, clock)

    assert result.outcome == "done"
    assert result.first_seen == {"preparing": 20.0, "transcribing": 40.0}
    assert result.elapsed_s == 50.0


def test_poll_job_stops_when_the_spend_cap_is_reached() -> None:
    clock = _Clock()
    session = _FakeSession([_status("running", "transcribing")])

    # $0.40/h with a $0.01 cap is reached after 90 s
    result = _poll(session, clock, max_spend=0.01)

    assert result.outcome == "spend-cap"
    assert 90.0 <= result.elapsed_s <= 100.0


def test_poll_job_reports_a_failed_job() -> None:
    clock = _Clock()
    session = _FakeSession([_status("failed", "labeling")])

    result = _poll(session, clock)

    assert result.outcome == "failed"
    assert "labeling" in result.first_seen


def test_parse_ffmpeg_duration() -> None:
    stderr = "Input #0, wav\n  Duration: 01:02:03.50, start: 0.000000, bitrate: 256 kb/s\n"

    assert parse_ffmpeg_duration(stderr) == 3723.5
    assert parse_ffmpeg_duration("nothing here") is None


def test_judge_reliability_filter() -> None:
    good = _offer()
    low = replace(_offer(), id=8, reliability=0.5)

    assert judge_reliability_filter([good], [good, low], 0.97)[0] == "PASS"
    assert judge_reliability_filter([good, low], [good, low], 0.97)[0] == "FAIL"
    assert judge_reliability_filter([], [good], 0.97)[0] == "FAIL"


def test_endpoint_tracker_summary() -> None:
    tracker = EndpointTracker()
    assert tracker.summary()[0] == "NOT RUN"
    for name in EndpointTracker.NEEDED:
        tracker.call(name, lambda: None)
    assert tracker.summary()[0] == "PASS"

    def boom() -> None:
        raise VastError("nope", status_code=500)

    with pytest.raises(VastError):
        tracker.call("create", boom)
    result, note = tracker.summary()
    assert result == "FAIL" and "500" in note
