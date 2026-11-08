"""Job data model and the on-disk job store."""

import dataclasses
import logging
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


def state_for_stage(stage: str) -> JobState:
    """Map a remote runner stage name to the job state shown in the app."""
    try:
        return _STAGE_TO_STATE[stage]
    except KeyError:
        raise ValueError(f"Unknown remote stage: {stage!r}") from None


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
        """Build a Job, ignoring unknown keys; missing optional keys take their defaults."""
        known = {f.name for f in dataclasses.fields(Job)}
        fields = {k: v for k, v in data.items() if k in known}
        fields["state"] = JobState(fields["state"])
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
        path = self.job_dir(job_id) / JOB_FILE
        data = read_json(path)
        if data is None:
            raise FileNotFoundError(f"No readable job file for job {job_id!r}")
        return self._parse(data)

    def load_all(self) -> list[Job]:
        """All readable jobs, newest first. Unreadable job files are skipped with a warning."""
        if not self._root.is_dir():
            return []
        jobs: list[Job] = []
        for path in self._root.glob(f"*/{JOB_FILE}"):
            data = read_json(path)
            try:
                if data is None:
                    raise ValueError("unreadable")
                jobs.append(self._parse(data))
            except (ValueError, TypeError, KeyError):
                logger.warning("Skipping unreadable job file %s", path.name)
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    @staticmethod
    def _parse(data: object) -> Job:
        if not isinstance(data, dict):
            raise ValueError("job file is not a JSON object")
        return Job.from_dict(data)
