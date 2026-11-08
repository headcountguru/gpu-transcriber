"""Background threads. Nothing slow or network-bound runs on the UI thread.

Workers never touch widgets: they report through signals, which Qt delivers on the UI thread.
No exception ever escapes a worker's ``run``: it is turned into a plain-words message. Only the
exception type is logged for unexpected errors, because a message could echo a secret.
"""

import logging
import threading
from collections.abc import Callable
from typing import Protocol

from PySide6.QtCore import QObject, QThread, Signal

from app.credentials import CredentialError
from app.hf_check import HfCheckError
from app.jobs import Job
from app.media_prep import MediaPrepError
from app.orchestrator import JobAlreadyRunningError
from app.vast_client import VastError
from app.ytdlp_tool import YtDlpError

logger = logging.getLogger(__name__)

# Exceptions whose message is already plain words and free of secrets.
PLAIN_ERRORS: tuple[type[Exception], ...] = (
    MediaPrepError,
    VastError,
    HfCheckError,
    YtDlpError,
    CredentialError,
    JobAlreadyRunningError,
)


class JobRunner(Protocol):
    """The part of ``Orchestrator`` a job worker needs."""

    def run(
        self, job_id: str, on_update: Callable[[Job], None], cancel: threading.Event | None = None
    ) -> Job: ...


def plain_error(exc: Exception) -> str:
    """A message safe to show: the text of a known plain error, else a generic sentence."""
    if isinstance(exc, PLAIN_ERRORS) and str(exc):
        return str(exc)
    return (
        f"Something went wrong ({type(exc).__name__}). "
        "Details are in the log file; please try again."
    )


def _log_failure(what: str, exc: Exception) -> None:
    if isinstance(exc, PLAIN_ERRORS):
        logger.warning("%s failed: %s", what, exc)
    else:
        logger.error("%s failed unexpectedly (%s)", what, type(exc).__name__)


class TaskWorker(QThread):
    """Runs ``task`` once on its own thread; emits ``succeeded(result)`` or ``failed(message)``."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        task: Callable[[], object],
        name: str = "Background task",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._task = task
        self._name = name

    def run(self) -> None:
        try:
            result = self._task()
        except Exception as exc:
            _log_failure(self._name, exc)
            self.failed.emit(plain_error(exc))
            return
        self.succeeded.emit(result)


class JobWorker(QThread):
    """Runs ``Orchestrator.run`` for one job.

    ``job_updated(Job)`` carries every saved change; ``ended(job_id, message)`` fires last, with
    an empty message on a normal finish or plain words if the run raised.
    """

    job_updated = Signal(object)
    ended = Signal(str, str)

    def __init__(self, runner: JobRunner, job_id: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._runner = runner
        self.job_id = job_id
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        """Ask the run to stop at its next check; the orchestrator shuts the machine down."""
        self.cancel_event.set()

    def run(self) -> None:
        message = ""
        try:
            self._runner.run(self.job_id, self.job_updated.emit, self.cancel_event)
        except Exception as exc:
            _log_failure(f"Job {self.job_id}", exc)
            message = plain_error(exc)
        self.ended.emit(self.job_id, message)
