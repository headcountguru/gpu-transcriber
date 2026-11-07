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


def max_runtime_s(duration_s: float) -> float:
    """Upper bound on how long a job for ``duration_s`` of audio may run."""
    return max(_RUNTIME_FLOOR_S, _RUNTIME_FACTOR * duration_s + _RUNTIME_OVERHEAD_S)


@dataclass(frozen=True)
class RemoteStatus:
    """Progress record the runner writes to ``status.json``.

    ``state`` is one of ``"running"``, ``"done"``, ``"failed"``.
    ``stage`` is one of ``"preparing"``, ``"transcribing"``, ``"labeling"``,
    ``"aligning"``, ``"writing"``, ``"done"``.
    """

    state: str
    stage: str
    percent: int
    message: str
    chunk: int | None = None
    chunks: int | None = None
    error: str | None = None
    updated_at: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(text: str) -> "RemoteStatus":
        return RemoteStatus(**json.loads(text))
