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

from app.jobs import TERMINAL_STATES, Job, JobState, JobStore, output_file_name
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

# Stall detection (see ``StallWatch``). The runner writes its first status.json within seconds of
# starting, before its heavy imports (live: about 10 s from launch); 15 minutes also covers a very
# slow disk. While running it rewrites the status at least every ``heartbeat_s`` (60 s), even in
# one long step with no progress (a 30-minute diarization chunk, forced alignment), so 20 minutes
# without any change is 20 missed heartbeats: the runner is dead, not slow. A runner that reports
# ``heartbeat_s`` 0 (an image built before the heartbeat) is never judged stale, because its silent
# steps have no known bound.
NO_FIRST_STATUS_S = 15 * 60.0
STALE_STATUS_S = 20 * 60.0
STALE_MIN_HEARTBEATS = 10  # never stricter than this many missed heartbeats
MSG_NEVER_REPORTED = (
    "The machine stopped reporting progress: the transcription never started reporting on it "
    "within {minutes:.0f} minutes, so the job was stopped. Finished parts are saved."
)
MSG_STOPPED_REPORTING = (
    "The machine stopped reporting progress for {minutes:.0f} minutes, so the job was stopped. "
    "Finished parts are saved."
)
MSG_NO_UNTRIED_OFFER = (
    "The only suitable machines available right now are ones that already failed for this job, "
    "so no other machine was rented. Finished parts are saved. Try again later."
)

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


class TriedOffers:
    """Offers, and the hosts behind them, already rented during one run of a job."""

    def __init__(self) -> None:
        self._offers: set[int] = set()
        self._hosts: set[int] = set()

    def remember(self, offer: Offer) -> None:
        self._offers.add(offer.id)
        if offer.machine_id is not None:
            self._hosts.add(offer.machine_id)

    def untried(self, offers: list[Offer]) -> list[Offer]:
        """``offers`` (order kept) without any tried offer or any offer on a tried host."""
        return [
            item
            for item in offers
            if item.id not in self._offers
            and (item.machine_id is None or item.machine_id not in self._hosts)
        ]


class StallWatch:
    """Tells a dead runner from a slow one, using only the app's own (injected) clock.

    The machine's ``updated_at`` is never compared with the PC's clock (they may disagree);
    only *whether it changed* matters, timed locally from when the watch saw it change.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._started = clock()
        self._seen: tuple[object, ...] | None = None
        self._changed_at = self._started
        self._heartbeat_s = 0.0

    def check(self, status: RemoteStatus | None) -> str | None:
        """None while all is well, else the plain-words reason the runner looks dead."""
        now = self._clock()
        if status is not None:
            seen = (status.state, status.stage, status.percent, status.message, status.chunk)
            seen += (status.updated_at,)
            if seen != self._seen:
                self._seen, self._changed_at = seen, now
                self._heartbeat_s = status.heartbeat_s if status.state == "running" else 0.0
                return None
        if self._seen is None:
            if now - self._started >= NO_FIRST_STATUS_S:
                return MSG_NEVER_REPORTED.format(minutes=NO_FIRST_STATUS_S / 60)
            return None
        if not isinstance(self._heartbeat_s, (int, float)) or self._heartbeat_s <= 0:
            return None
        limit = max(STALE_STATUS_S, STALE_MIN_HEARTBEATS * float(self._heartbeat_s))
        if now - self._changed_at >= limit:
            return MSG_STOPPED_REPORTING.format(minutes=limit / 60)
        return None


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


def pull_outputs(
    session: RemoteSession, remote_dir: str, local_dir: Path, base_name: str
) -> list[str]:
    """Download the three output files as ``<base_name>-transcript.txt`` and so on.

    Returns the local names that are missing or empty afterwards. A file the machine does not
    have counts as missing; other link errors are raised.
    """
    problems: list[str] = []
    for name in OUTPUT_FILES:
        local_name = output_file_name(base_name, name)
        target = local_dir / local_name
        try:
            session.download(f"{remote_dir}/{name}", target)
        except FileNotFoundError:
            problems.append(local_name)
            continue
        if not target.is_file() or target.stat().st_size == 0:
            problems.append(local_name)
    return problems


def destroy_and_confirm(
    vast: VastLike,
    instance_id: int,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    label_prefix: str,
) -> bool:
    """Destroy the instance (three tries), then confirm Vast.ai no longer lists it.

    A 404 from destroy means it is already gone. False means it may still be billing.
    Gone is confirmed only when the instance lookup says gone AND the listing of
    ``label_prefix`` machines no longer shows it (Vast.ai can be briefly inconsistent).
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
    return _confirm_gone(vast, instance_id, clock, sleep, label_prefix)


def _confirm_gone(
    vast: VastLike,
    instance_id: int,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    label_prefix: str,
) -> bool:
    deadline = clock() + CONFIRM_GONE_TIMEOUT_S
    while True:
        try:
            if vast.get_instance(instance_id) is None and not any(
                item.id == instance_id for item in vast.list_instances(label_prefix)
            ):
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


def mark_lost(store: JobStore, job: Job, destroyed: bool = True) -> None:
    """Fail an unfinished job whose machine is gone (or that never rented one).

    ``destroyed=False`` means its saved machine could not be confirmed destroyed: the id and
    start time are kept (the "may still be running" alarm) so ``retry_unconfirmed_destroys``
    keeps trying, and the message says so.
    """
    never_rented = job.instance_id is None and job.machine_started_at is None
    message = CLOSED_BEFORE_RENTING if never_rented else LOST_WHILE_CLOSED
    cost = max(job.banked_cost, job.cost_so_far)
    if not destroyed and job.instance_id is not None:
        warned = f"{message} {still_running_text(job.instance_id)}"
        store.save(job.with_(state=JobState.FAILED, message=warned, error=warned))
        return
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
