"""Job-folder layout and status schema shared by the runner and the desktop app."""

import json
from dataclasses import asdict, dataclass

JOB_FILE = "job.json"
JOB_INPUT_DIR = "input"
JOB_CHECKPOINT_DIR = "checkpoints"
JOB_OUTPUT_DIR = "output"
STATUS_FILE = "status.json"
OUTPUT_FILES = ("transcript.txt", "transcript_timestamped.txt", "transcript.srt")

GRACE_AFTER_FINISH_S = 10800.0

_RUNTIME_FLOOR_S = 10800.0
_RUNTIME_FACTOR = 0.75
_RUNTIME_OVERHEAD_S = 7200.0


# Error text from the machine (e.g. ffmpeg output) is shortened to this many characters, keeping
# both ends: the start says what failed, the end usually says why.
MAX_REMOTE_TEXT_CHARS = 400
_ELLIPSIS = "…"


def max_runtime_s(duration_s: float) -> float:
    """Upper bound on how long a job for ``duration_s`` of audio may run."""
    return max(_RUNTIME_FLOOR_S, _RUNTIME_FACTOR * duration_s + _RUNTIME_OVERHEAD_S)


def shorten_remote_text(text: str, limit: int = MAX_REMOTE_TEXT_CHARS) -> str:
    """``text`` if it fits in ``limit`` characters, else its start and end joined by "…".

    Scrub secrets *before* shortening: a secret cut in half would no longer be found.
    """
    if len(text) <= limit:
        return text
    keep = limit - len(_ELLIPSIS)
    head = keep - keep // 2
    return text[:head] + _ELLIPSIS + text[len(text) - keep // 2 :]


@dataclass(frozen=True)
class RemoteStatus:
    """Progress record the runner writes to ``status.json``.

    ``state`` is one of ``"running"``, ``"done"``, ``"failed"``.
    ``stage`` is one of ``"preparing"``, ``"transcribing"``, ``"labeling"``,
    ``"aligning"``, ``"writing"``, ``"done"``.
    ``heartbeat_s``: while ``running``, the runner rewrites this file at least this often (with a
    new ``updated_at``) even during a long step that reports no progress. 0 means a runner that
    does not, so its silence says nothing about whether it is alive.
    """

    state: str
    stage: str
    percent: int
    message: str
    chunk: int | None = None
    chunks: int | None = None
    error: str | None = None
    updated_at: float = 0.0
    heartbeat_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(text: str) -> "RemoteStatus":
        return RemoteStatus(**json.loads(text))
