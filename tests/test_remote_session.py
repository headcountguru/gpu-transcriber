"""Unit tests for ``app.remote_session``: key handling, connect/retry loop and launch command.

Real SFTP/exec behaviour is exercised live in Task 9; these tests inject a fake paramiko client.
"""

import logging
import shlex
from pathlib import Path
from typing import Any

import paramiko
import pytest

from app.remote_session import SshRemoteSession, ensure_ssh_key

SECRET = "hf_SUPERSECRETTOKEN123"


class FakeStatusEvent:
    def wait(self, timeout: float | None = None) -> bool:
        return True


class FakeChannel:
    def __init__(self, exit_status: int = 0) -> None:
        self.exit_status = exit_status
        self.status_event = FakeStatusEvent()


class FakeStream:
    def __init__(self, text: str = "", exit_status: int = 0) -> None:
        self.channel = FakeChannel(exit_status)
        self._text = text

    def read(self) -> bytes:
        return self._text.encode()


class FakeSftpChannel:
    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout


class FakeSftpFile:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self) -> "FakeSftpFile":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._data


class FakeSftp:
    """In-memory SFTP: ``dirs`` is the set of existing dirs, ``files`` maps path -> bytes."""

    def __init__(self) -> None:
        self.closed = False
        self.dirs: set[str] = {"/"}
        self.files: dict[str, bytes] = {}
        self.made: list[str] = []
        self.put_calls: list[tuple[str, str, Any]] = []

    def get_channel(self) -> FakeSftpChannel:
        return FakeSftpChannel()

    def close(self) -> None:
        self.closed = True

    def stat(self, path: str) -> None:
        if path not in self.dirs and path not in self.files:
            raise FileNotFoundError(path)

    def mkdir(self, path: str) -> None:
        self.made.append(path)
        self.dirs.add(path)

    def put(self, local: str, remote: str, callback: Any = None) -> None:
        self.put_calls.append((local, remote, callback))
        if callback is not None:
            callback(10, 10)

    def get(self, remote: str, local: str) -> None:
        Path(local).write_bytes(self.files[remote])

    def listdir(self, path: str) -> list[str]:
        if path not in self.dirs:
            raise FileNotFoundError(path)
        prefix = path.rstrip("/") + "/"
        return [p[len(prefix) :] for p in self.files if p.startswith(prefix)]

    def open(self, path: str, mode: str) -> FakeSftpFile:
        if path not in self.files:
            raise FileNotFoundError(path)
        return FakeSftpFile(self.files[path])


class FakeClient:
    """Stands in for ``paramiko.SSHClient``; fails ``connect`` ``fail_times`` times first."""

    instances: list["FakeClient"] = []

    def __init__(self, fail_times: int = 0, exit_status: int = 0) -> None:
        self.fail_times = fail_times
        self.exit_status = exit_status
        self.connect_calls: list[dict[str, Any]] = []
        self.commands: list[str] = []
        self.policy: object = None
        self.closed = False
        self.sftp = FakeSftp()

    def set_missing_host_key_policy(self, policy: object) -> None:
        self.policy = policy

    def connect(self, **kwargs: Any) -> None:
        self.connect_calls.append(kwargs)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("connection refused")

    def open_sftp(self) -> FakeSftp:
        return self.sftp

    def exec_command(self, command: str, timeout: float | None = None) -> tuple[Any, Any, Any]:
        self.commands.append(command)
        return None, FakeStream(exit_status=self.exit_status), FakeStream("err text")

    def close(self) -> None:
        self.closed = True


class FakeClock:
    """Monotonic clock that only advances when ``sleep`` is called."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def _connect(client: FakeClient, clock: FakeClock, timeout_s: float = 300.0) -> SshRemoteSession:
    return SshRemoteSession.connect(
        "1.2.3.4",
        2222,
        "/keys/id_ed25519",
        timeout_s=timeout_s,
        sleep=clock.sleep,
        client_factory=lambda: client,  # type: ignore[arg-type, return-value]
        clock=clock,
    )


def test_ensure_ssh_key_generates_then_reuses(tmp_path: Path) -> None:
    key_path, public = ensure_ssh_key(tmp_path)

    assert key_path == tmp_path / "id_ed25519"
    assert key_path.is_file()
    assert public.startswith("ssh-ed25519 ")
    assert "\n" not in public
    assert (tmp_path / "id_ed25519.pub").read_text(encoding="utf-8").strip() == public
    # paramiko must be able to load what we wrote
    paramiko.Ed25519Key.from_private_key_file(str(key_path))

    private_before = key_path.read_bytes()
    mtime_before = key_path.stat().st_mtime_ns
    key_path_2, public_2 = ensure_ssh_key(tmp_path)

    assert key_path_2 == key_path
    assert public_2 == public
    assert key_path.read_bytes() == private_before
    assert key_path.stat().st_mtime_ns == mtime_before


def test_ensure_ssh_key_recreates_missing_public_file(tmp_path: Path) -> None:
    key_path, public = ensure_ssh_key(tmp_path)
    (tmp_path / "id_ed25519.pub").unlink()

    _, public_2 = ensure_ssh_key(tmp_path)

    assert public_2 == public
    assert (tmp_path / "id_ed25519.pub").is_file()
    assert key_path.is_file()


def test_connect_retries_until_ready() -> None:
    client = FakeClient(fail_times=2)
    clock = FakeClock()

    session = _connect(client, clock)

    assert len(clock.sleeps) == 2
    assert len(client.connect_calls) == 3
    call = client.connect_calls[-1]
    assert call["hostname"] == "1.2.3.4"
    assert call["port"] == 2222
    assert call["key_filename"] == "/keys/id_ed25519"
    assert call["timeout"] > 0 and call["banner_timeout"] > 0 and call["auth_timeout"] > 0
    assert call["look_for_keys"] is False and call["allow_agent"] is False
    assert isinstance(client.policy, paramiko.AutoAddPolicy)
    assert isinstance(session, SshRemoteSession)


def test_connect_raises_after_timeout() -> None:
    client = FakeClient(fail_times=10**6)
    clock = FakeClock()

    with pytest.raises(ConnectionError, match="reachable"):
        _connect(client, clock, timeout_s=30.0)

    assert clock.now >= 30.0 or len(client.connect_calls) >= 1
    assert client.closed


def test_connect_retries_authentication_errors() -> None:
    class AuthFlaky(FakeClient):
        def connect(self, **kwargs: Any) -> None:
            self.connect_calls.append(kwargs)
            if len(self.connect_calls) < 2:
                raise paramiko.AuthenticationException("key not yet installed")

    client = AuthFlaky()
    clock = FakeClock()

    _connect(client, clock)

    assert len(clock.sleeps) == 1


def test_launch_exports_env_and_detaches(caplog: pytest.LogCaptureFixture) -> None:
    client = FakeClient()
    session = _connect(client, FakeClock())

    with caplog.at_level(logging.DEBUG):
        session.launch("python3 /root/run_job.py --job /root/job", {"HF_TOKEN": SECRET, "A": "x y"})

    assert len(client.commands) == 1
    command = client.commands[0]
    assert "nohup python3 /root/run_job.py --job /root/job" in command
    assert command.rstrip().endswith("&")
    assert f"export HF_TOKEN={shlex.quote(SECRET)};" in command
    assert "export A='x y'" in command
    assert SECRET not in caplog.text
    assert "HF_TOKEN" in caplog.text  # names may be logged, values may not


def test_launch_quotes_hostile_env_values() -> None:
    client = FakeClient()
    session = _connect(client, FakeClock())

    session.launch("true", {"X": "a'; rm -rf /; echo '"})

    assert "export X='a'\"'\"'; rm -rf /; echo '\"'\"''" in client.commands[0]


def test_launch_rejects_bad_env_name() -> None:
    session = _connect(FakeClient(), FakeClock())

    with pytest.raises(ValueError, match="environment variable name"):
        session.launch("true", {"BAD NAME": "v"})


def test_launch_failure_does_not_leak_secret() -> None:
    client = FakeClient(exit_status=1)
    session = _connect(client, FakeClock())

    with pytest.raises(RuntimeError) as excinfo:
        session.launch("true", {"HF_TOKEN": SECRET})

    assert SECRET not in str(excinfo.value)


def test_close_closes_sftp_and_client() -> None:
    client = FakeClient()
    session = _connect(client, FakeClock())

    session.close()

    assert client.sftp.closed and client.closed


def test_upload_creates_parents_and_reports_progress(tmp_path: Path) -> None:
    client = FakeClient()
    session = _connect(client, FakeClock())
    local = tmp_path / "a.wav"
    local.write_bytes(b"x")
    progress: list[tuple[int, int]] = []
    client.sftp.dirs.add("/root")

    session.upload(
        local, "/root/job/input/a.wav", lambda sent, total: progress.append((sent, total))
    )

    assert client.sftp.made == ["/root/job", "/root/job/input"]
    assert client.sftp.put_calls[0][:2] == (str(local), "/root/job/input/a.wav")
    assert progress == [(10, 10)]


def test_download_creates_local_parent(tmp_path: Path) -> None:
    client = FakeClient()
    client.sftp.files["/root/out.txt"] = b"hello"
    session = _connect(client, FakeClock())

    session.download("/root/out.txt", tmp_path / "sub" / "out.txt")

    assert (tmp_path / "sub" / "out.txt").read_bytes() == b"hello"


def test_list_files_and_read_text_handle_missing() -> None:
    client = FakeClient()
    client.sftp.dirs.add("/root/job")
    client.sftp.files["/root/job/status.json"] = '{"stage": "é"}'.encode()
    session = _connect(client, FakeClock())

    assert session.list_files("/nope") == []
    assert session.list_files("/root/job") == ["status.json"]
    assert session.read_text("/root/job/status.json") == '{"stage": "é"}'
    assert session.read_text("/root/job/missing.json") is None
