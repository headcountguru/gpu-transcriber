"""The runner always leaves a final status, keeps it fresh while alive, and bounds error text."""

import sys
import threading
import time
from pathlib import Path

import pytest
import run_job as rj

from common.protocol import MAX_REMOTE_TEXT_CHARS, STATUS_FILE, shorten_remote_text
from tests.test_run_job import TOKEN, FakeSteps, make_job, read_status


def test_import_error_while_loading_the_pipeline_writes_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> rj.PipelineSteps:
        raise ImportError(f"libcudnn missing ({TOKEN})")

    monkeypatch.setattr(rj, "apply_compat_patches", lambda: None)
    monkeypatch.setattr(rj, "default_steps", broken)
    job = make_job(tmp_path)

    with pytest.raises(ImportError):
        rj.run_job(str(job), TOKEN)

    status = read_status(job)
    assert status.state == "failed"
    assert status.error is not None and "ImportError" in status.error
    assert TOKEN not in (job / STATUS_FILE).read_text(encoding="utf-8")


def test_compat_patch_crash_writes_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> None:
        raise AttributeError("torchaudio has no attribute AudioMetaData")

    monkeypatch.setattr(rj, "apply_compat_patches", broken)
    job = make_job(tmp_path)

    with pytest.raises(AttributeError):
        rj.run_job(str(job), TOKEN)

    assert read_status(job).state == "failed"


def test_keyboard_interrupt_writes_failed_status_and_main_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = FakeSteps()
    steps.diarize_error = KeyboardInterrupt()  # type: ignore[assignment]
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(rj, "apply_compat_patches", lambda: None)
    monkeypatch.setattr(rj, "default_steps", steps.as_pipeline_steps)
    job = make_job(tmp_path)

    assert rj.main(["--job-dir", str(job)]) == rj.EXIT_FAILED

    status = read_status(job)
    assert status.state == "failed"
    assert "KeyboardInterrupt" in (status.error or "")


def test_system_exit_writes_failed_status_and_propagates(tmp_path: Path) -> None:
    steps = FakeSteps()
    steps.transcribe_error = SystemExit(3)  # type: ignore[assignment]
    job = make_job(tmp_path)

    with pytest.raises(SystemExit):
        rj.run_job(str(job), TOKEN, steps=steps.as_pipeline_steps())

    assert read_status(job).state == "failed"


def test_long_error_text_is_shortened_on_the_machine(tmp_path: Path) -> None:
    steps = FakeSteps()
    steps.diarize_error = RuntimeError("start " + "x" * 5000 + " the real ffmpeg reason")
    job = make_job(tmp_path)

    with pytest.raises(RuntimeError):
        rj.run_job(str(job), TOKEN, steps=steps.as_pipeline_steps())

    error = read_status(job).error or ""
    assert len(error) <= MAX_REMOTE_TEXT_CHARS
    assert error.startswith("RuntimeError: start")
    assert error.endswith("the real ffmpeg reason")
    assert "…" in error


def test_token_cut_in_half_by_shortening_is_still_redacted(tmp_path: Path) -> None:
    steps = FakeSteps()
    padding = "y" * (MAX_REMOTE_TEXT_CHARS // 2 - 20)
    steps.diarize_error = RuntimeError(padding + TOKEN + "z" * 5000)
    job = make_job(tmp_path)

    with pytest.raises(RuntimeError):
        rj.run_job(str(job), TOKEN, steps=steps.as_pipeline_steps())

    text = (job / STATUS_FILE).read_text(encoding="utf-8")
    assert TOKEN[:10] not in text


def test_shorten_remote_text_keeps_short_text_and_both_ends_of_long_text() -> None:
    assert shorten_remote_text("short") == "short"
    long = "a" * 300 + "b" * 300
    short = shorten_remote_text(long)
    assert len(short) == MAX_REMOTE_TEXT_CHARS
    assert short.startswith("a") and short.endswith("b") and "…" in short


def test_status_records_the_heartbeat_interval(tmp_path: Path) -> None:
    job = make_job(tmp_path)

    rj.run_job(str(job), TOKEN, steps=FakeSteps().as_pipeline_steps())

    assert read_status(job).heartbeat_s == rj.HEARTBEAT_S


def test_heartbeat_refreshes_status_during_one_long_silent_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rj, "HEARTBEAT_S", 0.02)
    steps = FakeSteps()
    seen: list[float] = []
    beats = threading.Event()

    def slow_alignment(
        words: list[dict], audio_path: str, language: str, device: str = "cuda"
    ) -> list[dict]:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:  # one long step that never reports progress
            status = read_status(job)
            if status.updated_at not in seen:
                seen.append(status.updated_at)
            if len(seen) >= 3:
                beats.set()
                break
            time.sleep(0.01)
        return words

    steps.apply_forced_alignment = slow_alignment  # type: ignore[method-assign]
    job = make_job(tmp_path)

    rj.run_job(str(job), TOKEN, steps=steps.as_pipeline_steps())

    assert beats.is_set()
    assert read_status(job).state == "done"
    assert not [t for t in threading.enumerate() if t.name == rj.HEARTBEAT_THREAD_NAME]


def test_heartbeat_stops_after_a_failure(tmp_path: Path) -> None:
    steps = FakeSteps()
    steps.diarize_error = RuntimeError("boom")
    job = make_job(tmp_path)

    with pytest.raises(RuntimeError):
        rj.run_job(str(job), TOKEN, steps=steps.as_pipeline_steps())

    assert not [t for t in threading.enumerate() if t.name == rj.HEARTBEAT_THREAD_NAME]
    assert read_status(job).state == "failed"


# --- C8: a second runner on the same job never starts work ----------------------------------------


def test_second_runner_exits_without_touching_the_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job(tmp_path)
    original = '{"state": "running", "stage": "labeling", "percent": 70, "message": "first"}'
    (job / STATUS_FILE).write_text(original, encoding="utf-8")
    steps = FakeSteps()
    monkeypatch.setattr(rj, "_hold_runner_lock", lambda _job: None)

    with pytest.raises(rj.RunnerAlreadyRunningError):
        rj.run_job(str(job), TOKEN, steps=steps.as_pipeline_steps())

    assert (job / STATUS_FILE).read_text(encoding="utf-8") == original
    assert steps.transcribe_calls == 0


def test_main_returns_one_for_a_second_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(rj, "_hold_runner_lock", lambda _job: None)

    assert rj.main(["--job-dir", str(make_job(tmp_path))]) == rj.EXIT_FAILED


def test_runner_lock_is_held_for_the_run_and_released_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    released: list[bool] = []
    real = rj._hold_runner_lock

    def tracking(job: Path) -> object:
        release = real(job)
        assert release is not None
        if sys.platform != "win32":  # flock exists only where the runner really runs
            assert real(job) is None  # a second runner is refused while the first holds it
        return lambda: (released.append(True), release())  # type: ignore[misc]

    monkeypatch.setattr(rj, "_hold_runner_lock", tracking)
    job = make_job(tmp_path)

    rj.run_job(str(job), TOKEN, steps=FakeSteps().as_pipeline_steps())

    assert released == [True]
    if sys.platform != "win32":
        again = real(job)
        assert again is not None
        again()
