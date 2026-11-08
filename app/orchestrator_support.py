"""Helpers for the orchestrator: machine shutdown, label sweeps, cost, status parsing, downloads,
the commands run on the machine, and the app-start bookkeeping for unfinished jobs.

Kept apart from ``app.orchestrator`` so the job flow there stays readable. The shutdown and
sweep sequencing follows ``scripts/live_smoke.py`` (proven against the real Vast.ai on
2026-11-07): destroy is always called (three tries) and then confirmed by polling, and an
unclear create is followed by a bounded search for anything carrying the job's label.
"""

import logging
import re
import shlex
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import paramiko

from app.jobs import TERMINAL_STATES, Job, JobState, JobStore
from app.remote_session import RemoteSession
from app.settings import Settings
from app.vast_client import Instance, Offer, VastError
from common.protocol import OUTPUT_FILES, RemoteStatus

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600.0
DESTROY_ATTEMPTS = 3
DESTROY_RETRY_S = 5.0
CONFIRM_GONE_TIMEOUT_S = 90.0
CONFIRM_POLL_S = 5.0
CREATE_SWEEP_WINDOW_S = 180.0  # how long to look for a machine whose create was unclear
CREATE_SWEEP_INTERVAL_S = 10.0
HTTP_NOT_FOUND = 404
REDACTED = "***"
RUNNER_PYTHONPATH = "/app:/app/remote"  # as in the image (see scripts/live_smoke.py)
LOST_WHILE_CLOSED = "The machine was lost while the app was closed."
CLOSED_BEFORE_RENTING = "The app was closed before this job rented a machine. Start it again."

# The only checkpoint names ever downloaded or uploaded (see remote/chunker.py and
# remote/run_job.py). Anything else a machine lists, including names with path parts, is ignored.
CHECKPOINT_NAME_RE = re.compile(r"diar_\d+\.json|words\.json")

# Errors that mean "the SSH/SFTP link to the machine failed".
LINK_ERRORS: tuple[type[BaseException], ...] = (OSError, EOFError, paramiko.SSHException)


class VastLike(Protocol):
    """The parts of ``VastClient`` the orchestrator uses (a fake implements this in tests)."""

    def search_offers(self, settings: Settings | None = None) -> list[Offer]: ...

    def create_instance(
        self, offer_id: int, image: str, label: str, disk_gb: int, onstart: str, env: dict
    ) -> int: ...

    def get_instance(self, instance_id: int) -> Instance | None: ...

    def list_instances(self, label_prefix: str) -> list[Instance]: ...

    def destroy_instance(self, instance_id: int) -> None: ...

    def get_balance(self) -> float: ...

    def attach_ssh_key(self, instance_id: int, public_key: str) -> None: ...


def build_onstart(settings: Settings) -> str:
    """Start-up script: run the watchdog so a machine whose app died still destroys itself.

    The same text as ``onstart_script`` in ``scripts/live_smoke.py``, proven live.
    """
    job_dir = shlex.quote(settings.remote_job_dir)
    return (
        f"mkdir -p {job_dir}\n"
        f"nohup env PYTHONPATH={RUNNER_PYTHONPATH} python /app/remote/watchdog.py "
        f"--job-dir {job_dir} > /tmp/watchdog.log 2>&1 &\n"
    )


def runner_command(remote_job_dir: str) -> str:
    """The command that starts ``run_job.py`` (the token is passed in the environment)."""
    return (
        f"env PYTHONPATH={RUNNER_PYTHONPATH} python /app/remote/run_job.py "
        f"--job-dir {shlex.quote(remote_job_dir)}"
    )


def still_running_text(instance_id: int) -> str:
    return (
        f"The machine (Vast.ai instance {instance_id}) may still be running and charging: "
        "destroy it in the Vast.ai console."
    )


def scrub(text: str, secrets: Iterable[str]) -> str:
    """Replace every non-empty secret in ``text`` with ``***``."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return text


def machine_cost(
    banked: float, started_at: float | None, price_per_hour: float, now: float
) -> float:
    """Money spent so far: earlier machines plus the current one since its create call."""
    if started_at is None:
        return banked
    hours = max(0.0, now - started_at) / SECONDS_PER_HOUR
    return banked + hours * price_per_hour


def parse_status(text: str | None) -> RemoteStatus | None:
    """Parse ``status.json``; a missing, partial or corrupt file means "no status yet"."""
    if not text:
        return None
    try:
        status = RemoteStatus.from_json(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(status.state, str) or not isinstance(status.stage, str):
        return None
    if not isinstance(status.percent, int) or not isinstance(status.message, str):
        return None
    return status


def is_checkpoint_name(name: str) -> bool:
    return CHECKPOINT_NAME_RE.fullmatch(name) is not None


def safe_child(folder: Path, name: str) -> Path | None:
    """``folder / name`` only if ``name`` is a plain file name that stays inside ``folder``."""
    if not name or name in {".", ".."} or any(ch in name for ch in "/\\:\0"):
        return None
    candidate = (folder / name).resolve()
    if candidate.parent != folder.resolve():
        return None
    return candidate


def local_checkpoints(folder: Path) -> list[Path]:
    """Checkpoint files already on the PC for one job, by name."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and is_checkpoint_name(p.name))


def pull_checkpoints(session: RemoteSession, remote_dir: str, local_dir: Path) -> list[str]:
    """Download checkpoint files not yet on the PC. Returns the names downloaded.

    Raises a ``LINK_ERRORS`` error if the machine cannot be reached.
    """
    have = {p.name for p in local_checkpoints(local_dir)}
    pulled: list[str] = []
    for name in sorted(session.list_files(remote_dir)):
        if name in have or not is_checkpoint_name(name):
            continue
        target = safe_child(local_dir, name)
        if target is None:
            continue
        session.download(f"{remote_dir}/{name}", target)
        pulled.append(name)
    return pulled


def pull_outputs(session: RemoteSession, remote_dir: str, local_dir: Path) -> list[str]:
    """Download the three output files; return the names missing or empty afterwards.

    A file the machine does not have counts as missing; other link errors are raised.
    """
    problems: list[str] = []
    for name in OUTPUT_FILES:
        target = local_dir / name
        try:
            session.download(f"{remote_dir}/{name}", target)
        except FileNotFoundError:
            problems.append(name)
            continue
        if not target.is_file() or target.stat().st_size == 0:
            problems.append(name)
    return problems


def destroy_and_confirm(
    vast: VastLike, instance_id: int, clock: Callable[[], float], sleep: Callable[[float], None]
) -> bool:
    """Destroy the instance (three tries), then confirm Vast.ai no longer lists it.

    A 404 from destroy means it is already gone. False means it may still be billing.
    """
    for attempt in range(1, DESTROY_ATTEMPTS + 1):
        try:
            vast.destroy_instance(instance_id)
        except VastError as exc:
            if exc.status_code == HTTP_NOT_FOUND:
                logger.info("Instance %s was already gone", instance_id)
                break
            logger.warning(
                "Destroy attempt %d for instance %s failed: %s", attempt, instance_id, exc
            )
            if attempt < DESTROY_ATTEMPTS:
                sleep(DESTROY_RETRY_S)
        else:
            break
    return _confirm_gone(vast, instance_id, clock, sleep)


def _confirm_gone(
    vast: VastLike, instance_id: int, clock: Callable[[], float], sleep: Callable[[float], None]
) -> bool:
    deadline = clock() + CONFIRM_GONE_TIMEOUT_S
    while True:
        try:
            if vast.get_instance(instance_id) is None:
                logger.info("Confirmed instance %s is gone", instance_id)
                return True
        except VastError as exc:
            logger.warning("Could not confirm instance %s is gone: %s", instance_id, exc)
        if clock() >= deadline:
            logger.critical("Instance %s could not be confirmed destroyed", instance_id)
            return False
        sleep(CONFIRM_POLL_S)


def instances_with_label(vast: VastLike, label: str) -> list[Instance] | None:
    """Instances whose label is exactly ``label``, or None if the listing failed."""
    try:
        listed = vast.list_instances(label)
    except VastError as exc:
        logger.error("Could not list instances labelled %s: %s", label, exc)
        return None
    return [item for item in listed if item.label == label]


def sweep_label(
    vast: VastLike, label: str, clock: Callable[[], float], sleep: Callable[[float], None]
) -> set[int] | None:
    """After an unclear create, look for a machine with ``label`` for a bounded time.

    A new instance may not be listed at once, so one look is not enough. Returns the ids found
    (empty if nothing appeared), or None if the label could never be listed.
    """
    deadline = clock() + CREATE_SWEEP_WINDOW_S
    listed_once = False
    while True:
        found = instances_with_label(vast, label)
        if found:
            return {item.id for item in found}
        listed_once = listed_once or found is not None
        if clock() >= deadline:
            return set() if listed_once else None
        sleep(CREATE_SWEEP_INTERVAL_S)


# --- app start -----------------------------------------------------------------------------------


def reattach(
    store: JobStore,
    job: Job,
    found: list[Instance],
    clock: Callable[[], float],
    destroy: Callable[[int], bool],
) -> None:
    """Point an unfinished job at its machine (found by label); destroy any second one."""
    keep = next((item for item in found if item.id == job.instance_id), found[0])
    for extra in found:
        if extra.id != keep.id:
            logger.warning("Destroying a second machine %s for job %s", extra.id, job.id)
            destroy(extra.id)
    started = job.machine_started_at
    if started is None:
        # The app died between saving the job and saving the create time. Vast.ai's instance
        # record (as parsed) has no start time, so assume billing ran since the job was made:
        # this over-counts rather than under-counts against the cap.
        started = job.created_at
        logger.warning(
            "Job %s had no machine start time; counting cost from when the job was created",
            job.id,
        )
    store.save(
        job.with_(
            instance_id=keep.id,
            machine_started_at=started,
            price_per_hour=job.price_per_hour or keep.dph_total,
            message="Reconnecting to the machine",
        )
    )
    logger.info("Job %s re-attached to machine %s", job.id, keep.id)


def settled_job(job: Job, cost: float) -> Job:
    """A finished job whose machine is now confirmed gone: clear its machine and the warning."""
    warning = still_running_text(job.instance_id) if job.instance_id is not None else ""

    def without_warning(text: str) -> str:
        return text.replace(warning, "").strip() if warning else text

    error = without_warning(job.error) if job.error else None
    return job.with_(
        instance_id=None,
        machine_started_at=None,
        banked_cost=cost,
        cost_so_far=cost,
        message=without_warning(job.message),
        error=error or None,
    )


def mark_lost(store: JobStore, job: Job) -> None:
    """Fail an unfinished job whose machine is gone (or that never rented one)."""
    never_rented = job.instance_id is None and job.machine_started_at is None
    message = CLOSED_BEFORE_RENTING if never_rented else LOST_WHILE_CLOSED
    cost = max(job.banked_cost, job.cost_so_far)
    store.save(
        job.with_(
            state=JobState.FAILED,
            message=message,
            error=message,
            instance_id=None,
            machine_started_at=None,
            banked_cost=cost,
            cost_so_far=cost,
        )
    )


def belongs_to_unfinished_job(store: JobStore, job_id: str) -> bool:
    """True only for a readable job that has not finished; unknown or corrupt jobs are False."""
    try:
        job = store.get(job_id)
    except (FileNotFoundError, ValueError):
        return False
    return job.state not in TERMINAL_STATES
