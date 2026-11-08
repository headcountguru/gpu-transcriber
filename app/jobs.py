"""Job data model and the on-disk job store."""

import dataclasses
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from common.jsonio import read_json, write_json_atomic

logger = logging.getLogger(__name__)

JOB_FILE = "job.json"
JOB_SUBDIRS = ("audio", "checkpoints", "output")


class JobState(str, Enum):
    """Where a job is in its life, from first preparation to a final outcome."""

    PREPARING = "PREPARING"
    UPLOADING = "UPLOADING"
    STARTING_MACHINE = "STARTING_MACHINE"
    TRANSCRIBING = "TRANSCRIBING"
    LABELING = "LABELING"
    SAVING = "SAVING"
    DONE = "DONE"
    FAILED = "FAILED"
    STOPPED_AT_CAP = "STOPPED_AT_CAP"
    CANCELLED = "CANCELLED"


TERMINAL_STATES: frozenset[JobState] = frozenset(
    {JobState.DONE, JobState.FAILED, JobState.STOPPED_AT_CAP, JobState.CANCELLED}
)

_STAGE_TO_STATE: dict[str, JobState] = {
    "preparing": JobState.STARTING_MACHINE,
    "transcribing": JobState.TRANSCRIBING,
    "labeling": JobState.LABELING,
    "aligning": JobState.LABELING,
    "writing": JobState.SAVING,
    "done": JobState.SAVING,
}


def output_file_name(base_name: str, remote_name: str) -> str:
    """Name a finished file after its recording: ``<title>-transcript.txt`` and so on.

    ``remote_name`` is what the machine calls the file (``transcript.txt``,
    ``transcript_timestamped.txt``, ``transcript.srt``); the title never has a media extension.
    """
    return f"{base_name}-{remote_name}"


def state_for_stage(stage: str) -> JobState:
    """Map a remote runner stage name to the job state shown in the app."""
    try:
        return _STAGE_TO_STATE[stage]
    except KeyError:
        raise ValueError(f"Unknown remote stage: {stage!r}") from None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return _is_int(value) or (isinstance(value, float) and math.isfinite(value))


# Declared type of every Job field except ``state``: (check, description, may be None).
_FIELD_CHECKS: dict[str, tuple[Callable[[object], bool], str, bool]] = {
    "id": (lambda v: isinstance(v, str), "a string", False),
    "name": (lambda v: isinstance(v, str), "a string", False),
    "source": (lambda v: isinstance(v, str), "a string", False),
    "is_youtube": (lambda v: isinstance(v, bool), "a boolean", False),
    "created_at": (_is_number, "a finite number", False),
    "percent": (_is_int, "an integer", False),
    "message": (lambda v: isinstance(v, str), "a string", False),
    "chunk": (_is_int, "an integer", True),
    "chunks": (_is_int, "an integer", True),
    "audio_path": (lambda v: isinstance(v, str), "a string", True),
    "duration_s": (_is_number, "a finite number", True),
    "instance_id": (_is_int, "an integer", True),
    "price_per_hour": (_is_number, "a finite number", False),
    "machine_started_at": (_is_number, "a finite number", True),
    "banked_cost": (_is_number, "a finite number", False),
    "cost_so_far": (_is_number, "a finite number", False),
    "cost_cap": (_is_number, "a finite number", False),
    "output_dir": (lambda v: isinstance(v, str), "a string", True),
    "error": (lambda v: isinstance(v, str), "a string", True),
    "retries": (_is_int, "an integer", False),
}
_REQUIRED_FIELDS = ("id", "name", "source", "is_youtube", "state", "created_at")


@dataclass(frozen=True)
class Job:
    """One transcription job. Immutable; use ``with_`` to get an updated copy."""

    id: str
    name: str
    source: str
    is_youtube: bool
    state: JobState
    created_at: float
    percent: int = 0
    message: str = ""
    chunk: int | None = None
    chunks: int | None = None
    audio_path: str | None = None
    duration_s: float | None = None
    instance_id: int | None = None
    price_per_hour: float = 0.0
    machine_started_at: float | None = None
    banked_cost: float = 0.0
    cost_so_far: float = 0.0
    cost_cap: float = 0.0
    output_dir: str | None = None
    error: str | None = None
    retries: int = 0

    def with_(self, **changes: object) -> "Job":
        """Return a copy with ``changes`` applied; unknown field names raise TypeError."""
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        data = dataclasses.asdict(self)
        data["state"] = self.state.value
        return data

    @staticmethod
    def from_dict(data: dict[str, object]) -> "Job":
        """Build a Job, ignoring unknown keys; missing optional keys take their defaults.

        Raises ValueError if a required key is missing or any field has the wrong type.
        """
        for key in _REQUIRED_FIELDS:
            if key not in data:
                raise ValueError(f"Missing required job field: {key}")
        fields: dict[str, object] = {}
        for key, (is_valid, description, may_be_none) in _FIELD_CHECKS.items():
            if key not in data:
                continue
            value = data[key]
            if value is None and may_be_none:
                fields[key] = None
            elif is_valid(value):
                fields[key] = value
            else:
                raise ValueError(f"Job field {key} must be {description}")
        state = data["state"]
        if not isinstance(state, str) or state not in JobState._value2member_map_:
            raise ValueError(f"Unknown job state: {state!r}")
        fields["state"] = JobState(state)
        return Job(**fields)  # type: ignore[arg-type]


class JobStore:
    """Stores each job as ``<root>/<job_id>/job.json`` next to its working folders."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def job_dir(self, job_id: str) -> Path:
        """Folder for ``job_id`` (holds ``audio/``, ``checkpoints/``, ``output/``).

        Pure path computation: nothing is created here; ``save`` creates the folders.
        """
        if (
            not job_id
            or job_id in {".", ".."}
            or any(ch in job_id for ch in ("/", "\\", ":", "\0"))
        ):
            raise ValueError(f"Invalid job id: {job_id!r}")
        return self._root / job_id

    def save(self, job: Job) -> None:
        job_dir = self.job_dir(job.id)
        for sub in JOB_SUBDIRS:
            (job_dir / sub).mkdir(parents=True, exist_ok=True)
        write_json_atomic(job_dir / JOB_FILE, job.to_dict())

    def get(self, job_id: str) -> Job:
        """Load one job.

        Raises FileNotFoundError if the job has no job file, and ValueError if the file exists
        but is unreadable or invalid (bad JSON, wrong shape, wrong-typed or unknown values).
        """
        path = self.job_dir(job_id) / JOB_FILE
        if not path.exists():
            raise FileNotFoundError(f"No job file for job {job_id!r}")
        return self._parse(read_json(path))

    def load_all(self) -> list[Job]:
        """All readable jobs, newest first. Unreadable job files are skipped with a warning."""
        if not self._root.is_dir():
            return []
        jobs: list[Job] = []
        for path in self._root.glob(f"*/{JOB_FILE}"):
            try:
                jobs.append(self._parse(read_json(path)))
            except ValueError as exc:
                logger.warning("Skipping unreadable job %s: %s", path.parent.name, exc)
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    @staticmethod
    def _parse(data: object) -> Job:
        if not isinstance(data, dict):
            raise ValueError("job file is missing, unparseable or not a JSON object")
        return Job.from_dict(data)
