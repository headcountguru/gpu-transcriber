"""Start the desktop app: ``python -m app.main`` (or ``--selftest`` to check the install).

Builds the orchestrator from its parts, asks for the keys once if they are missing, opens the
window, and lets the window run the start-up checks on a worker thread.
"""

import argparse
import logging
import os
import sys
import threading
from collections.abc import Callable, Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from app.credentials import HF_TOKEN, VAST_API_KEY, CredentialError, CredentialStore
from app.jobs import Job, JobStore
from app.media_prep import (
    MediaPrepError,
    PreparedAudio,
    prepare_file,
    prepare_youtube,
    probe_duration_s,
)
from app.remote_session import RemoteSession, SshRemoteSession, check_private_key, ensure_ssh_key
from app.settings import IMAGE, Settings, app_data_dir
from app.vast_client import Instance, VastClient
from app.ytdlp_tool import YtDlpError, ensure_ytdlp

logger = logging.getLogger("app")

JOBS_DIR = "jobs"
TOOLS_DIR = "tools"
LOGS_DIR = "logs"
LOG_FILE = "gpu_transcriber.log"
LOG_MAX_BYTES = 1_000_000
LOG_BACKUPS = 3
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
AUDIO_DIR = "audio"
# How long to keep trying SSH on a freshly started machine (as in scripts/live_smoke.py).
SSH_CONNECT_TIMEOUT_S = 300.0

FfmpegPath = Callable[[], str]
YtdlpPath = Callable[[], Path]


def configure_logging(log_dir: Path) -> None:
    """Log to a rotating file in ``log_dir`` and to the console. Secrets are never logged."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler = RotatingFileHandler(
        log_dir / LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
    )
    console = logging.StreamHandler()
    for handler in (file_handler, console):
        handler.setFormatter(formatter)
        root.addHandler(handler)


def ffmpeg_path() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def ytdlp_getter(tools_dir: Path) -> YtdlpPath:
    """Return a function giving the yt-dlp path, downloading it on first use.

    A lock stops two threads (the start-up download and a length check) downloading at once.
    """
    lock = threading.Lock()

    def get() -> Path:
        with lock:
            return ensure_ytdlp(tools_dir)

    return get


def make_prepare(
    store: JobStore, ffmpeg: FfmpegPath, ytdlp: YtdlpPath
) -> Callable[[Job], PreparedAudio]:
    """``prepare(job)``: download or convert the job's audio into its ``audio`` folder."""

    def prepare(job: Job) -> PreparedAudio:
        work_dir = store.job_dir(job.id) / AUDIO_DIR
        if not job.is_youtube:
            return prepare_file(Path(job.source), work_dir, ffmpeg())
        try:
            ytdlp_exe = ytdlp()
        except YtDlpError as exc:
            raise MediaPrepError(str(exc)) from None
        return prepare_youtube(job.source, work_dir, ffmpeg(), ytdlp_exe)

    return prepare


def make_probe(ffmpeg: FfmpegPath, ytdlp: YtdlpPath) -> Callable[[str, bool], float]:
    """``probe(source, is_youtube)``: the recording's length in seconds (worker thread only)."""

    def probe(source: str, is_youtube: bool) -> float:
        if not is_youtube:
            return probe_duration_s(source, False, ffmpeg(), Path())
        try:
            ytdlp_exe = ytdlp()
        except YtDlpError as exc:
            raise MediaPrepError(str(exc)) from None
        return probe_duration_s(source, True, ffmpeg(), ytdlp_exe)

    return probe


def make_session_factory(
    key_path: Path,
) -> Callable[[Instance, Callable[[float], None]], RemoteSession]:
    """``session_factory(instance, sleep)``: an SSH session to a running machine.

    ``sleep`` is the job's own wait between connect retries: it raises when the job is
    cancelled or reaches its spending cap, which ends the retries at once. A missing or damaged
    key file raises ``SshKeyError`` before any connect (retrying could never help).
    """

    def factory(instance: Instance, sleep: Callable[[float], None]) -> RemoteSession:
        if not instance.ssh_host or instance.ssh_port is None:
            raise ConnectionError("The machine has no SSH address yet.")
        check_private_key(key_path)
        return SshRemoteSession.connect(
            instance.ssh_host,
            instance.ssh_port,
            str(key_path),
            timeout_s=SSH_CONNECT_TIMEOUT_S,
            sleep=sleep,
        )

    return factory


class CredentialVast:
    """A ``VastClient`` that always uses the currently saved key (it may change in Settings)."""

    def __init__(self, creds: CredentialStore) -> None:
        self._creds = creds

    def __repr__(self) -> str:
        return "CredentialVast()"

    def _client(self) -> Any:
        return VastClient(self._creds.get(VAST_API_KEY) or "")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client(), name)


def keys_present(creds: CredentialStore) -> bool:
    try:
        return bool(creds.get(VAST_API_KEY)) and bool(creds.get(HF_TOKEN))
    except CredentialError as exc:
        logger.warning("%s", exc)
        return False


def keyring_usable(get_backend: Callable[[], object] | None = None) -> bool:
    """True if keyring found a real secure-storage backend (its ``fail`` backend means none).

    Only asks keyring which backend it would use; it never reads or writes a credential.
    """
    try:
        from keyring.backends import fail

        if get_backend is None:
            import keyring

            get_backend = keyring.get_keyring
        return not isinstance(get_backend(), fail.Keyring)
    except Exception as exc:  # a missing package or backend must fail the self-test, not crash it
        logger.error("Secure storage check failed (%s)", type(exc).__name__)
        return False


def selftest() -> int:
    """Import everything, build the Qt app offscreen (no window), check ffmpeg and keyring."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import app.gui.main_window  # noqa: F401
    import app.gui.setup_dialog  # noqa: F401
    import app.orchestrator  # noqa: F401

    application = QApplication.instance() or QApplication([])
    ffmpeg = Path(ffmpeg_path())
    if not ffmpeg.is_file():
        logger.error("Self-test failed: ffmpeg was not found at %s", ffmpeg)
        return 1
    if not keyring_usable():
        logger.error("Self-test failed: no secure storage (Windows Credential Manager) backend")
        return 1
    logger.info("Self-test passed (Qt %s, ffmpeg %s)", application.applicationName(), ffmpeg)
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gpu-transcriber", description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="check the install and exit")
    return parser.parse_args(argv)


def build_orchestrator(data_dir: Path, store: JobStore, creds: CredentialStore) -> Any:
    """The real orchestrator: Vast.ai with the saved key, SSH with the app's own key pair."""
    from app.orchestrator import Orchestrator

    key_path, public_key = ensure_ssh_key(data_dir)
    return Orchestrator(
        Settings(image=IMAGE),
        store,
        creds,
        CredentialVast(creds),
        make_session_factory(key_path),
        make_prepare(store, ffmpeg_path, ytdlp_getter(data_dir / TOOLS_DIR)),
        ssh_public_key=public_key,
    )


def run_app(data_dir: Path) -> int:
    """Open the window and run the Qt event loop. Returns the exit code."""
    from PySide6.QtWidgets import QApplication, QDialog, QWidget

    from app.gui import theme
    from app.gui.main_window import MainWindow
    from app.gui.setup_dialog import SetupDialog
    from app.gui.worker import TaskWorker
    from app.hf_check import check_hf_access

    application = QApplication(sys.argv[:1])
    application.setApplicationName("GPU Transcriber")
    application.setStyleSheet(theme.STYLESHEET)
    creds = CredentialStore()

    def make_setup(parent: QWidget | None) -> QDialog:
        return SetupDialog(
            creds,
            check_vast=lambda key: VastClient(key).get_balance(),
            check_hf=check_hf_access,
            parent=parent,
        )

    if not keys_present(creds) and make_setup(None).exec() != QDialog.DialogCode.Accepted:
        logger.info("Setup was closed without saving keys; exiting")
        return 0
    for sub in (JOBS_DIR, TOOLS_DIR):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    store = JobStore(data_dir / JOBS_DIR)
    ytdlp = ytdlp_getter(data_dir / TOOLS_DIR)
    window = MainWindow(
        build_orchestrator(data_dir, store, creds),
        store,
        creds,
        probe=make_probe(ffmpeg_path, ytdlp),
        setup_dialog_factory=make_setup,
    )
    warm = TaskWorker(ytdlp, "yt-dlp download", window)  # never blocks start-up
    warm.failed.connect(window.show_tools_problem)  # its own slot: never hides other warnings
    warm.start()
    window.show()
    code = int(application.exec())
    if window.has_background_work() or warm.isRunning():
        # A job (or a check) is still running on a worker thread. The user agreed to close: the
        # rented machine carries on by itself and the job resumes on the next start. Exit hard,
        # because Qt aborts if a QThread object is destroyed while it is still running.
        logger.info("Closing while background work is still running")
        logging.shutdown()
        os._exit(code)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
        return selftest()
    data_dir = app_data_dir()
    configure_logging(data_dir / LOGS_DIR)
    return run_app(data_dir)


if __name__ == "__main__":
    sys.exit(main())
