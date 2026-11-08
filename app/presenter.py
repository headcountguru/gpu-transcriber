"""Turns jobs and user input into the plain words the window shows. No Qt here."""

from dataclasses import dataclass

from app.jobs import TERMINAL_STATES, Job, JobState
from app.media_prep import MediaPrepError, check_youtube_url

DONE = "done"
ACTIVE = "active"
PENDING = "pending"
FAILED = "failed"

STAGE_PREPARING = "Preparing"
STAGE_STARTING = "Starting machine"
STAGE_UPLOADING = "Uploading"
STAGE_TRANSCRIBING = "Transcribing"
STAGE_LABELING = "Labeling speakers"
STAGE_SAVING = "Saving results"

# Display order follows what the orchestrator actually does: a machine is rented and started
# before the audio is uploaded to it.
_STAGE_STATES = (
    JobState.PREPARING,
    JobState.STARTING_MACHINE,
    JobState.UPLOADING,
    JobState.TRANSCRIBING,
    JobState.LABELING,
    JobState.SAVING,
)
_STAGE_INDEX = {state: index for index, state in enumerate(_STAGE_STATES)}
_LABELING_INDEX = _STAGE_INDEX[JobState.LABELING]
_SAVING_INDEX = _STAGE_INDEX[JobState.SAVING]
_TRANSCRIBING_INDEX = _STAGE_INDEX[JobState.TRANSCRIBING]
_STARTING_INDEX = _STAGE_INDEX[JobState.STARTING_MACHINE]

_FINAL_TEXT = {
    JobState.DONE: "Done",
    JobState.FAILED: "Failed",
    JobState.STOPPED_AT_CAP: "Stopped at the spending cap",
    JobState.CANCELLED: "Cancelled",
}

_STATUS_KIND = {
    JobState.DONE: "done",
    JobState.FAILED: "failed",
    JobState.STOPPED_AT_CAP: "cap",
    JobState.CANCELLED: "muted",
}

MSG_EMPTY_INPUT = "Paste a YouTube link or choose an audio file."
CAP_STEP = 0.01
CAP_SUGGESTION_MULTIPLE = 2.0
_LINK_PREFIXES = ("http://", "https://", "www.", "youtube.com", "youtu.be", "m.youtube.com")


@dataclass(frozen=True)
class StageRow:
    """One line in the stage list. ``status`` is ``done``, ``active``, ``pending`` or ``failed``."""

    label: str
    status: str


@dataclass(frozen=True)
class SourceCheck:
    """What the new-job input holds.

    ``ok`` means Play can start it now (a valid link). ``needs_file_check`` means it is a path:
    whether the file exists is checked on a worker thread, never here, because a network share
    or a sleeping mapped drive can block for seconds.
    """

    source: str
    is_youtube: bool
    ok: bool
    error: str | None
    needs_file_check: bool = False


def _labeling_label(job: Job) -> str:
    if job.chunk is not None and job.chunks is not None:
        return f"{STAGE_LABELING} (chunk {job.chunk} of {job.chunks})"
    return STAGE_LABELING


def _stage_labels(job: Job) -> list[str]:
    return [
        STAGE_PREPARING,
        STAGE_STARTING,
        STAGE_UPLOADING,
        STAGE_TRANSCRIBING,
        _labeling_label(job),
        STAGE_SAVING,
    ]


def _stopped_stage_index(job: Job) -> int:
    """For a finished job, the stage it was in when it stopped (inferred from saved fields)."""
    if job.output_dir:
        return _SAVING_INDEX
    if job.chunk is not None:
        return _LABELING_INDEX
    if job.percent > 0:
        return _TRANSCRIBING_INDEX
    started = job.machine_started_at is not None or job.instance_id is not None
    if job.audio_path or started or job.banked_cost > 0:
        return _STARTING_INDEX
    return 0


def stage_rows(job: Job) -> list[StageRow]:
    """The six stages in order, each marked done, active, pending or failed."""
    labels = _stage_labels(job)
    if job.state is JobState.DONE:
        return [StageRow(label, DONE) for label in labels]
    if job.state in TERMINAL_STATES:
        current = _stopped_stage_index(job)
        marker = FAILED if job.state is JobState.FAILED else PENDING
    else:
        current = _STAGE_INDEX[job.state]
        marker = ACTIVE
    rows: list[StageRow] = []
    for index, label in enumerate(labels):
        if index < current:
            rows.append(StageRow(label, DONE))
        elif index == current:
            rows.append(StageRow(label, marker))
        else:
            rows.append(StageRow(label, PENDING))
    return rows


def cost_text(job: Job) -> str:
    """``$0.21 of $2.00 cap``, or ``$0.00 so far`` before a cap has been chosen."""
    if job.cost_cap > 0:
        return f"${job.cost_so_far:.2f} of ${job.cost_cap:.2f} cap"
    return f"${job.cost_so_far:.2f} so far"


def summary_line(job: Job) -> str:
    """Short status for the job list and the detail header."""
    final = _FINAL_TEXT.get(job.state)
    if final is not None:
        return final
    label = _stage_labels(job)[_STAGE_INDEX[job.state]]
    if job.percent > 0:
        return f"{label} · {job.percent}%"
    return label


def status_kind(job: Job) -> str:
    """Color family for a job's status: ``running``, ``done``, ``failed``, ``cap`` or ``muted``."""
    return _STATUS_KIND.get(job.state, "running")


def alarm_text(job: Job) -> str | None:
    """Warning for a finished job whose machine was never confirmed gone, else ``None``."""
    if job.state not in TERMINAL_STATES or job.instance_id is None:
        return None
    return (
        f"A machine may still be running and billing — id {job.instance_id} — the app is "
        "retrying to remove it. You can also check it yourself at cloud.vast.ai."
    )


def suggested_cap(job: Job) -> tuple[float, float]:
    """``(minimum, default)`` in dollars for "Raise cap and continue"."""
    spent = max(job.cost_so_far, job.banked_cost)
    minimum = round(max(job.cost_cap, spent) + CAP_STEP, 2)
    default = round(max(minimum, job.cost_cap * CAP_SUGGESTION_MULTIPLE), 2)
    return minimum, default


def _looks_like_link(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith(_LINK_PREFIXES) or "://" in lowered


def check_source(text: str) -> SourceCheck:
    """Classify the new-job input as a YouTube link, a path to check, or a plain error.

    Text only: this never touches the filesystem (it runs on every keystroke).
    """
    cleaned = text.strip().strip('"').strip()
    if not cleaned:
        return SourceCheck(source="", is_youtube=False, ok=False, error=None)
    if _looks_like_link(cleaned):
        try:
            link = check_youtube_url(cleaned)
        except MediaPrepError as exc:
            return SourceCheck(source=cleaned, is_youtube=True, ok=False, error=str(exc))
        return SourceCheck(source=link, is_youtube=True, ok=True, error=None)
    if "\\" in cleaned or "/" in cleaned or ":" in cleaned:
        return SourceCheck(
            source=cleaned, is_youtube=False, ok=False, error=None, needs_file_check=True
        )
    return SourceCheck(source=cleaned, is_youtube=False, ok=False, error=MSG_EMPTY_INPUT)
