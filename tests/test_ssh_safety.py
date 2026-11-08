"""A damaged SSH key fails fast (and is replaced at start); connect waits honour cancel and cap."""

import threading
from collections.abc import Callable
from pathlib import Path

import paramiko
import pytest

import app.main as app_main
from app.jobs import JobState
from app.remote_session import KEY_NAME, SshKeyError, check_private_key, ensure_ssh_key
from app.vast_client import Instance
from tests.test_orchestrator import FakeSession, World
from tests.test_orchestrator import world as world  # noqa: F401  (the fixture)

INSTANCE = Instance(id=1, status="running", ssh_host="h", ssh_port=22, dph_total=0.2, label="x")


# --- C1: a damaged key file ---------------------------------------------------------------------


@pytest.mark.parametrize("content", [b"", b"not a key at all", b"-----BEGIN OPENSSH PRIVATE KEY"])
def test_ensure_ssh_key_replaces_an_empty_or_damaged_key(tmp_path: Path, content: bytes) -> None:
    (tmp_path / KEY_NAME).write_bytes(content)
    (tmp_path / f"{KEY_NAME}.pub").write_text("ssh-ed25519 AAAAstale\n", encoding="ascii")

    key_path, public = ensure_ssh_key(tmp_path)

    paramiko.Ed25519Key.from_private_key_file(str(key_path))  # loadable now
    assert public.startswith("ssh-ed25519 ") and public != "ssh-ed25519 AAAAstale"
    assert (tmp_path / f"{KEY_NAME}.pub").read_text(encoding="ascii").strip() == public
    assert (tmp_path / f"{KEY_NAME}.bad").read_bytes() == content  # kept for a look


def test_ensure_ssh_key_fixes_a_public_file_that_does_not_match(tmp_path: Path) -> None:
    _, public = ensure_ssh_key(tmp_path)
    (tmp_path / f"{KEY_NAME}.pub").write_text("ssh-ed25519 AAAAother\n", encoding="ascii")

    _, again = ensure_ssh_key(tmp_path)

    assert again == public


@pytest.mark.parametrize("content", [None, b"", b"garbage"], ids=["missing", "empty", "garbage"])
def test_check_private_key_rejects_a_missing_or_damaged_key(
    tmp_path: Path, content: bytes | None
) -> None:
    path = tmp_path / KEY_NAME
    if content is not None:
        path.write_bytes(content)

    with pytest.raises(SshKeyError) as err:
        check_private_key(path)

    assert "restart" in str(err.value).lower()


def test_session_factory_fails_fast_on_a_damaged_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = tmp_path / KEY_NAME
    key.write_bytes(b"")
    connects: list[object] = []
    monkeypatch.setattr(
        app_main.SshRemoteSession, "connect", lambda *a, **k: connects.append(a) or "session"
    )

    with pytest.raises(SshKeyError):
        app_main.make_session_factory(key)(INSTANCE, lambda _s: None)

    assert connects == []


def test_session_factory_passes_the_jobs_sleep_to_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path, _ = ensure_ssh_key(tmp_path)
    seen: dict[str, object] = {}

    def fake_connect(host: str, port: int, key: str, **kwargs: object) -> str:
        seen.update(kwargs)
        return "session"

    def job_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(app_main.SshRemoteSession, "connect", fake_connect)

    assert app_main.make_session_factory(key_path)(INSTANCE, job_sleep) == "session"
    assert seen["sleep"] is job_sleep


def test_damaged_key_fails_the_job_in_plain_words_and_destroys_the_machine(
    world: World,
) -> None:
    def broken(_instance: Instance, _sleep: Callable[[float], None]) -> FakeSession:
        raise SshKeyError("The app's SSH key file is damaged. Restart the app to make a new one.")

    world.orch._session_factory = broken  # type: ignore[assignment]

    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert "SSH key" in (job.error or "") and "Something went wrong" not in (job.error or "")
    assert world.vast.instances == {}


# --- C6: cancel and cap during the SSH connect retries --------------------------------------------


def _slow_connect(
    world: World, on_sleep: Callable[[int], None], step_s: float = 5.0
) -> list[float]:
    sleeps: list[float] = []

    def factory(instance: Instance, sleep: Callable[[float], None]) -> FakeSession:
        for _ in range(60):  # sshd not answering for a long time
            sleep(step_s)
            sleeps.append(step_s)
            on_sleep(len(sleeps))
        return world.session_factory(instance)

    world.orch._session_factory = factory  # type: ignore[assignment]
    return sleeps


def test_cancel_during_the_ssh_connect_stops_at_once(world: World) -> None:
    cancel = threading.Event()
    sleeps = _slow_connect(world, lambda n: cancel.set() if n == 2 else None)

    job = world.submit_and_run(cancel=cancel)

    assert job.state is JobState.CANCELLED
    assert len(sleeps) == 2
    assert world.vast.instances == {}


def test_spending_cap_during_the_ssh_connect_stops_at_the_cap(world: World) -> None:
    sleeps = _slow_connect(world, lambda _n: None, step_s=60.0)

    job = world.submit_and_run(cap=1.0)  # $0.001 a second: the cap is reached after ~1000 s

    assert job.state is JobState.STOPPED_AT_CAP
    assert len(sleeps) < 60
    assert world.vast.instances == {}
