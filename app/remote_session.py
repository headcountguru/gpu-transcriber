"""SSH/SFTP session to the rented GPU machine, plus the app's SSH key management.

SECRETS: ``launch`` passes the Hugging Face token to the remote process through the environment
(``export NAME='value'`` in the exec channel). The value is never written to a file on the machine,
never logged (only variable *names* are), and never put in an exception message.

Every network operation carries a timeout so a dead machine cannot hang the app.
"""

import getpass
import logging
import os
import posixpath
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import paramiko
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)

KEY_NAME = "id_ed25519"
SSH_USER = "root"
RETRY_INTERVAL_S = 5.0
ATTEMPT_TIMEOUT_S = 15.0  # per TCP connect / SSH banner / auth attempt
CHANNEL_TIMEOUT_S = 60.0  # SFTP and exec channel operations
LAUNCH_LOG_PATH = "/tmp/gpu_transcriber_runner.log"  # single-tenant rented machine
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Errors that mean "sshd not answering (yet)": refused/unreachable/timeouts, dropped handshake,
# and key-not-yet-installed (a fresh instance may take a moment to apply the registered key).
_RETRYABLE = (OSError, EOFError, paramiko.SSHException)


class RemoteSession(Protocol):
    """What the orchestrator needs from the remote machine."""

    def upload(
        self,
        local: Path,
        remote: str,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None: ...

    def download(self, remote: str, local: Path) -> None: ...

    def list_files(self, remote_dir: str) -> list[str]: ...

    def read_text(self, remote_path: str) -> str | None: ...

    def launch(self, command: str, env: dict[str, str]) -> None: ...

    def close(self) -> None: ...


def _restrict_to_current_user(path: Path) -> None:
    """Best effort: make ``path`` readable only by the current user. Never raises."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"{getpass.getuser()}:F"],
                check=True,
                capture_output=True,
                timeout=15,
            )
        else:
            os.chmod(path, 0o600)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not restrict permissions on %s: %s", path, exc)


def _public_text(private_key: ed25519.Ed25519PrivateKey) -> str:
    raw = private_key.public_key().public_bytes(
        serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
    )
    return raw.decode("ascii")


def ensure_ssh_key(app_dir: Path) -> tuple[Path, str]:
    """Return ``(private_key_path, public_key_text)``, generating an ed25519 pair once.

    The pair lives in ``app_dir`` as ``id_ed25519`` / ``id_ed25519.pub`` and is reused afterwards.
    paramiko cannot generate ed25519 keys, so ``cryptography`` does it; the private key is written
    in unencrypted OpenSSH format, which paramiko loads.
    """
    private_path = app_dir / KEY_NAME
    public_path = app_dir / f"{KEY_NAME}.pub"
    if private_path.is_file():
        if public_path.is_file():
            return private_path, public_path.read_text(encoding="ascii").strip()
        loaded = serialization.load_ssh_private_key(private_path.read_bytes(), password=None)
        if not isinstance(loaded, ed25519.Ed25519PrivateKey):
            raise ValueError(f"{private_path} is not an ed25519 key")
        public = _public_text(loaded)
        public_path.write_text(public + "\n", encoding="ascii")
        return private_path, public

    app_dir.mkdir(parents=True, exist_ok=True)
    key = ed25519.Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    )
    private_path.touch()
    _restrict_to_current_user(private_path)  # restrict before the secret is written
    private_path.write_bytes(pem)
    public = _public_text(key)
    public_path.write_text(public + "\n", encoding="ascii")
    logger.info("Generated new ed25519 SSH key at %s", private_path)
    return private_path, public


class SshRemoteSession:
    """``RemoteSession`` implemented with paramiko (SSH exec + SFTP)."""

    def __init__(self, client: paramiko.SSHClient, sftp: paramiko.SFTPClient) -> None:
        self._client = client
        self._sftp = sftp

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        key_filename: str,
        timeout_s: float = 300.0,
        sleep: Callable[[float], None] = time.sleep,
        *,
        client_factory: Callable[[], paramiko.SSHClient] = paramiko.SSHClient,
        clock: Callable[[], float] = time.monotonic,
    ) -> "SshRemoteSession":
        """Connect, retrying until sshd answers or ``timeout_s`` passes.

        Raises ``ConnectionError`` ("never became reachable") when the deadline is reached.
        """
        deadline = clock() + timeout_s
        attempts = 0
        while True:
            attempts += 1
            client = client_factory()
            # Trade-off: freshly rented machines have unknown host keys and we do not persist
            # known_hosts. Machines are ephemeral and addressed through the provider's own API, so
            # we accept the first-seen key. A later task may pin it per instance.
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    hostname=host,
                    port=port,
                    username=SSH_USER,
                    key_filename=key_filename,
                    timeout=ATTEMPT_TIMEOUT_S,
                    banner_timeout=ATTEMPT_TIMEOUT_S,
                    auth_timeout=ATTEMPT_TIMEOUT_S,
                    look_for_keys=False,
                    allow_agent=False,
                )
                sftp = client.open_sftp()
                sftp.get_channel().settimeout(CHANNEL_TIMEOUT_S)
            except _RETRYABLE as exc:
                client.close()
                if clock() >= deadline:
                    raise ConnectionError(
                        f"The machine at {host}:{port} never became reachable over SSH "
                        f"after {attempts} attempts in {timeout_s:.0f}s (last error: {exc})"
                    ) from exc
                logger.info("SSH not ready on %s:%s (attempt %d): %s", host, port, attempts, exc)
                sleep(RETRY_INTERVAL_S)
                continue
            logger.info("SSH connected to %s:%s after %d attempt(s)", host, port, attempts)
            return cls(client, sftp)

    def upload(
        self,
        local: Path,
        remote: str,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self._mkdir_p(posixpath.dirname(remote))
        self._sftp.put(str(local), remote, callback=on_progress)

    def download(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._sftp.get(remote, str(local))

    def list_files(self, remote_dir: str) -> list[str]:
        try:
            return list(self._sftp.listdir(remote_dir))
        except FileNotFoundError:
            return []

    def read_text(self, remote_path: str) -> str | None:
        try:
            with self._sftp.open(remote_path, "rb") as handle:
                return bytes(handle.read()).decode("utf-8")
        except FileNotFoundError:
            return None

    def launch(self, command: str, env: dict[str, str]) -> None:
        """Start ``command`` detached (``nohup``, survives disconnect) with ``env`` exported.

        Returns once the remote shell has backgrounded the command, not when it finishes.
        """
        for name in env:
            if not _ENV_NAME_RE.match(name):
                raise ValueError(f"Invalid environment variable name: {name!r}")
        exports = "".join(f"export {name}={shlex.quote(value)}; " for name, value in env.items())
        remote_command = f"{exports}nohup {command} > {LAUNCH_LOG_PATH} 2>&1 < /dev/null &"
        # Log the command only, never the exported values.
        logger.info("Launching on machine: %s (env: %s)", command, ", ".join(sorted(env)))
        _, stdout, _ = self._client.exec_command(remote_command, timeout=CHANNEL_TIMEOUT_S)
        channel = stdout.channel
        if not channel.status_event.wait(CHANNEL_TIMEOUT_S):
            raise TimeoutError("Timed out waiting for the remote shell to start the job")
        if channel.exit_status != 0:
            raise RuntimeError(f"Remote launch failed with exit status {channel.exit_status}")

    def close(self) -> None:
        try:
            self._sftp.close()
        finally:
            self._client.close()

    def _mkdir_p(self, remote_dir: str) -> None:
        """Create ``remote_dir`` and any missing parents (SFTP has no ``mkdir -p``)."""
        if remote_dir in ("", "/", "."):
            return
        built = "/" if remote_dir.startswith("/") else ""
        for part in remote_dir.split("/"):
            if not part:
                continue
            built = posixpath.join(built, part)
            try:
                self._sftp.stat(built)
            except FileNotFoundError:
                self._sftp.mkdir(built)
