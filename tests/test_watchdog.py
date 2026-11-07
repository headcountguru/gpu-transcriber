import logging
import subprocess
from pathlib import Path

import pytest
from watchdog import (
    destroy_self,
    main,
    read_finished_at,
    read_max_runtime_s,
    should_self_destruct,
)

from common.protocol import JOB_FILE, STATUS_FILE, RemoteStatus


def test_destroys_after_grace_once_finished():
    assert should_self_destruct(
        now=1000, started_at=0, finished_at=100, grace_s=800, max_runtime_s=99999
    )


def test_waits_during_grace():
    assert not should_self_destruct(
        now=500, started_at=0, finished_at=100, grace_s=800, max_runtime_s=99999
    )


def test_destroys_after_max_runtime_even_if_unfinished():
    assert should_self_destruct(
        now=5000, started_at=0, finished_at=None, grace_s=800, max_runtime_s=4000
    )


def test_waits_when_unfinished_and_under_max_runtime():
    assert not should_self_destruct(
        now=3000, started_at=0, finished_at=None, grace_s=800, max_runtime_s=4000
    )


def test_read_max_runtime_falls_back_to_default(tmp_path: Path):
    assert read_max_runtime_s(tmp_path, 10800.0) == 10800.0


def test_read_max_runtime_reads_job_json(tmp_path: Path):
    (tmp_path / JOB_FILE).write_text('{"max_runtime_s": 5400}', encoding="utf-8")
    assert read_max_runtime_s(tmp_path, 10800.0) == 5400.0


@pytest.mark.parametrize(
    "text",
    ['{"max_runtime_s": "soon"}', '{"max_runtime_s": -5}', '{"max_runtime_s": true}', "[1]", "{"],
)
def test_read_max_runtime_ignores_bad_values(tmp_path: Path, text: str):
    (tmp_path / JOB_FILE).write_text(text, encoding="utf-8")
    assert read_max_runtime_s(tmp_path, 10800.0) == 10800.0


def _write_status(job_dir: Path, state: str, updated_at: float) -> None:
    status = RemoteStatus(
        state=state, stage="done", percent=100, message="m", updated_at=updated_at
    )
    (job_dir / STATUS_FILE).write_text(status.to_json(), encoding="utf-8")


@pytest.mark.parametrize("state", ["done", "failed"])
def test_finished_at_is_updated_at_for_terminal_states(tmp_path: Path, state: str):
    _write_status(tmp_path, state, 123.5)
    assert read_finished_at(tmp_path) == 123.5


def test_finished_at_none_while_running(tmp_path: Path):
    _write_status(tmp_path, "running", 123.5)
    assert read_finished_at(tmp_path) is None


@pytest.mark.parametrize(
    "text",
    [
        None,
        "{not json",
        "[1, 2]",
        '{"state": "done"}',
        '{"state": "done", "stage": "x", "percent": 1, "message": "m", "bogus": 1}',
        '{"state": "done", "stage": "x", "percent": 1, "message": "m", "updated_at": "late"}',
    ],
)
def test_finished_at_none_for_missing_or_bad_status(tmp_path: Path, text: str | None):
    if text is not None:
        (tmp_path / STATUS_FILE).write_text(text, encoding="utf-8")
    assert read_finished_at(tmp_path) is None


def test_destroy_self_passes_container_credentials():
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    destroy_self(run=fake_run, env={"CONTAINER_ID": "42", "CONTAINER_API_KEY": "k"})

    assert calls == [["vastai", "destroy", "instance", "42", "--api-key", "k"]]


@pytest.mark.parametrize(
    "env", [{}, {"CONTAINER_ID": "42"}, {"CONTAINER_API_KEY": "k"}, {"CONTAINER_ID": ""}]
)
def test_destroy_self_raises_when_env_missing(env: dict[str, str]):
    def fake_run(args, **kwargs):
        raise AssertionError("must not run")

    with pytest.raises(RuntimeError, match="CONTAINER_ID"):
        destroy_self(run=fake_run, env=env)


def test_destroy_self_raises_without_leaking_key_when_command_fails():
    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr="bad key SECRETKEY")

    with pytest.raises(RuntimeError) as exc:
        destroy_self(run=fake_run, env={"CONTAINER_ID": "42", "CONTAINER_API_KEY": "SECRETKEY"})

    assert "SECRETKEY" not in str(exc.value)
    assert exc.value.__cause__ is None


def test_destroy_self_raises_when_vastai_missing():
    def fake_run(args, **kwargs):
        raise FileNotFoundError("vastai")

    with pytest.raises(RuntimeError, match="vastai"):
        destroy_self(run=fake_run, env={"CONTAINER_ID": "42", "CONTAINER_API_KEY": "k"})


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def test_main_loop_destroys_exactly_once_then_returns_zero(tmp_path: Path):
    clock = FakeClock()
    destroyed: list[float] = []

    def fake_destroy() -> None:
        destroyed.append(clock.t)

    # Job finishes at t=120 (runner's updated_at); grace is 10800s; runtime limit is far off.
    (tmp_path / JOB_FILE).write_text('{"max_runtime_s": 999999}', encoding="utf-8")
    _write_status(tmp_path, "done", 120.0)

    code = main(
        ["--job-dir", str(tmp_path)],
        clock=clock.now,
        sleep=clock.sleep,
        destroy=fake_destroy,
    )

    assert code == 0
    assert len(destroyed) == 1
    assert destroyed[0] >= 120.0 + 10800.0
    assert all(s == 60.0 for s in clock.sleeps)


def test_main_destroys_at_max_runtime_when_job_never_finishes(tmp_path: Path):
    clock = FakeClock()
    destroyed: list[float] = []
    (tmp_path / JOB_FILE).write_text('{"max_runtime_s": 300}', encoding="utf-8")

    code = main(
        ["--job-dir", str(tmp_path)],
        clock=clock.now,
        sleep=clock.sleep,
        destroy=lambda: destroyed.append(clock.t),
    )

    assert code == 0
    assert destroyed == [300.0]


def test_main_logs_failure_and_retries_next_poll(tmp_path: Path, caplog):
    clock = FakeClock()
    attempts: list[float] = []
    (tmp_path / JOB_FILE).write_text('{"max_runtime_s": 120}', encoding="utf-8")

    def flaky_destroy() -> None:
        attempts.append(clock.t)
        if len(attempts) < 3:
            raise RuntimeError("vastai destroy failed")

    with caplog.at_level(logging.ERROR):
        code = main(
            ["--job-dir", str(tmp_path)],
            clock=clock.now,
            sleep=clock.sleep,
            destroy=flaky_destroy,
        )

    assert code == 0
    assert attempts == [120.0, 180.0, 240.0]
    assert "vastai destroy failed" in caplog.text


def test_main_uses_default_runtime_when_job_json_absent(tmp_path: Path):
    clock = FakeClock()
    destroyed: list[float] = []

    main(
        ["--job-dir", str(tmp_path)],
        clock=clock.now,
        sleep=clock.sleep,
        destroy=lambda: destroyed.append(clock.t),
    )

    assert destroyed == [10800.0]
