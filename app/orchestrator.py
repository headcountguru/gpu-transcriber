"""Drives one transcription job end to end: prepare, rent, upload, run, poll, download, destroy.

The window runs ``Orchestrator.run`` on a worker thread and shows each saved ``Job``.

Billing safety, in order of importance:
  - A job is saved (its label is ``<label_prefix><job id>``) before a machine is rented, and
    the instance id is saved as soon as Vast.ai returns it, so a crash cannot lose track of it.
  - Every way out of a machine (done, failed, cap, cancel, lost, a bug) destroys it and polls
    until Vast.ai confirms it is gone. If that cannot be confirmed, the job says so with its id.
  - A create with an unclear outcome is followed by a bounded search for the job's label, and
    anything found is destroyed. A definite refusal skips the search.
  - The spending cap is a hard limit, checked at every wait, upload callback and poll.
  - The machine's start-up script runs the on-machine watchdog, so a machine whose app died
    still destroys itself.
  - The same job is never run twice at once (a second ``run`` raises ``JobAlreadyRunningError``),
    so it can never rent two machines.

Invariant: a job in a final state whose ``instance_id`` is still set means "the destroy was not
confirmed, so the machine may still be running". Its ``machine_started_at`` is kept too, so its
cost keeps counting and its cap keeps being enforced. ``retry_unconfirmed_destroys`` (also run
by ``reap_orphans``) settles such jobs once Vast.ai confirms the machine is gone.

The cap can be overshot by a bounded amount: up to one poll interval, the time to download
checkpoints and outputs, the destroy and its confirmation (up to about 100 s), the SSH connect
retries inside ``session_factory`` (not interruptible from here), and the 180 s label sweep
after an unclear create.

Secrets: the Vast.ai key and Hugging Face token are scrubbed from every message and error saved
on a job and from every log line written here. The token reaches the machine only through the
environment of the launched runner.
"""

import logging
import math
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path, PureWindowsPath

from app.credentials import HF_TOKEN, VAST_API_KEY, CredentialError, CredentialStore
from app.estimate import default_cost_cap, estimate_job
from app.jobs import TERMINAL_STATES, Job, JobState, JobStore, state_for_stage
from app.media_prep import UNTITLED, MediaPrepError, PreparedAudio, safe_folder_name
from app.orchestrator_support import (
    LINK_ERRORS,
    VastLike,
    belongs_to_unfinished_job,
    build_onstart,
    destroy_and_confirm,
    local_checkpoints,
    machine_cost,
    mark_lost,
    parse_status,
    pull_checkpoints,
    pull_outputs,
    reattach,
    runner_command,
    scrub,
    settled_job,
    still_running_text,
    sweep_label,
)
from app.remote_session import RemoteSession
from app.settings import Settings
from app.vast_client import Instance, Offer, VastAmbiguousCreateError, VastError, rank_offers
from common.jsonio import write_json_atomic
from common.protocol import (
    JOB_CHECKPOINT_DIR,
    JOB_FILE,
    JOB_INPUT_DIR,
    JOB_OUTPUT_DIR,
    STATUS_FILE,
    RemoteStatus,
    max_runtime_s,
)

logger = logging.getLogger(__name__)

ATTACH_ATTEMPTS = 6  # a new instance may refuse the SSH key for about a minute
ATTACH_RETRY_S = 10.0
STARTUP_TIMEOUT_S = 1200.0
MAX_UNREACHABLE_POLLS = 6
STARTUP_BUDGET_HOURS = 0.25  # the cap must at least pay for a 15-minute startup
DEFINITE_REFUSAL_CODES = frozenset({400, 401, 403, 404, 422})
DEAD_INSTANCE_STATUSES = frozenset({"exited", "offline", "unknown"})
RUNNER_TOKEN_ENV = "HF_TOKEN"
LOCAL_CHECKPOINT_DIR = "checkpoints"  # JobStore's per-job folders
LOCAL_OUTPUT_DIR = "output"
REMOTE_SPEC_NAME = "remote_job.json"  # local copy of the job.json sent to the machine

MSG_NO_OFFER = "No suitable machine is available right now. Nothing was rented. Try again later."
MSG_NO_TOKEN = 'Add your Hugging Face token under "Accounts and keys…" before starting a job.'
MSG_CANCELLED = "Cancelled. Nothing more will be charged for this job."
MSG_DONE = "Done. The transcript is saved."
MSG_ALREADY_RUNNING = "This job is already running."
MSG_KEY_REFUSED = (
    "Could not set up access to the rented machine: Vast.ai kept refusing the app's SSH key."
)


class JobAlreadyRunningError(RuntimeError):
    """``run`` was called for a job that is already running; nothing was done."""


class _Stop(Exception):
    """Ends work on the current machine; the subclass says why."""


class _Cancelled(_Stop):
    pass


class _CapReached(_Stop):
    pass


class _MachineLost(_Stop):
    """The machine died or cannot be reached; the job may continue on a replacement."""


class _JobFailed(_Stop):
    """The job cannot continue. The message is plain words for the user."""


class _JobRun:
    """The job being run, saved (scrubbed) and reported after every change."""

    def __init__(
        self, store: JobStore, job: Job, on_update: Callable[[Job], None], cancel: threading.Event
    ) -> None:
        self.store = store
        self.job = job
        self.on_update = on_update
        self.cancel = cancel
        self.secrets: list[str] = []

    def save(self, **changes: object) -> Job:
        for key in ("message", "error"):
            value = changes.get(key)
            if isinstance(value, str):
                changes[key] = scrub(value, self.secrets)
        self.job = self.job.with_(**changes)
        self.store.save(self.job)
        try:
            self.on_update(self.job)
        except Exception as exc:  # a broken window callback must not strand a machine
            logger.warning("The job update callback failed: %s", type(exc).__name__)
        return self.job


class Orchestrator:
    """Runs jobs on rented Vast.ai machines, resuming, capping and cleaning up as needed."""

    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        creds: CredentialStore,
        vast: VastLike,
        session_factory: Callable[[Instance], RemoteSession],
        prepare: Callable[[Job], PreparedAudio],
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        *,
        ssh_public_key: str,
        poll_interval_s: float | None = None,
        max_machine_retries: int | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._creds = creds
        self._vast = vast
        self._session_factory = session_factory
        self._prepare = prepare
        self._clock = clock
        self._sleep = sleep
        self._public_key = ssh_public_key
        self._poll_interval_s = (
            settings.poll_interval_s if poll_interval_s is None else poll_interval_s
        )
        self._max_retries = (
            settings.max_machine_retries if max_machine_retries is None else max_machine_retries
        )
        self._running: set[str] = set()
        self._running_lock = threading.Lock()

    # --- public API -----------------------------------------------------------------------------

    def submit(self, source: str, is_youtube: bool, name: str | None = None) -> Job:
        """Save a new PREPARING job. Its name is safe to use as a Windows folder name."""
        if name:
            title = name
        elif is_youtube:
            title = UNTITLED  # replaced by the video's title once it is downloaded
        else:
            title = PureWindowsPath(source).stem
        job = Job(
            id=uuid.uuid4().hex,
            name=safe_folder_name(title),
            source=source,
            is_youtube=is_youtube,
            state=JobState.PREPARING,
            created_at=self._clock(),
            message="Waiting to start",
        )
        self._store.save(job)
        return job

    def run(
        self, job_id: str, on_update: Callable[[Job], None], cancel: threading.Event | None = None
    ) -> Job:
        """Drive the job to a final state (blocking). Returns the final job.

        Raises ``JobAlreadyRunningError`` (a ``RuntimeError``) if this job is already being run,
        so the window can say "This job is already running" instead of renting a second machine.
        """
        with self._running_lock:
            if job_id in self._running:
                raise JobAlreadyRunningError(MSG_ALREADY_RUNNING)
            self._running.add(job_id)
        try:
            return self._run_claimed(job_id, on_update, cancel or threading.Event())
        finally:
            with self._running_lock:
                self._running.discard(job_id)

    def _run_claimed(
        self, job_id: str, on_update: Callable[[Job], None], cancel: threading.Event
    ) -> Job:
        ctx = _JobRun(self._store, self._store.get(job_id), on_update, cancel)
        if ctx.job.state in TERMINAL_STATES:
            return ctx.job
        try:
            hf_token = self._load_secrets(ctx)
            self._ensure_prepared(ctx)
            self._run_machines(ctx, hf_token)
        except _JobFailed as exc:
            message = scrub(str(exc), ctx.secrets)
            logger.warning("Job %s failed: %s", job_id, message)
            self._settle(ctx, JobState.FAILED, message, error=message)
        except _CapReached:
            cap = ctx.job.cost_cap
            message = (
                f"Stopped at the spending cap of ${cap:.2f}. Finished parts are saved; "
                "raise the cap to continue."
            )
            self._settle(ctx, JobState.STOPPED_AT_CAP, message)
        except _Cancelled:
            self._settle(ctx, JobState.CANCELLED, MSG_CANCELLED)
        except Exception as exc:  # a bug must never leave a machine billing
            text = scrub(f"{type(exc).__name__}: {exc}", ctx.secrets)
            logger.error("Job %s stopped unexpectedly: %s", job_id, text)
            message = f"Something went wrong and the job was stopped ({text})."
            self._settle(ctx, JobState.FAILED, message, error=message)
        return ctx.job

    def resume_pending(self) -> list[str]:
        """At app start: re-attach unfinished jobs to their machines, found by label.

        Returns the ids of re-attached jobs (call ``run`` for each). A job whose machine is not
        found is marked FAILED. Unreadable job files are skipped. A failed listing raises
        ``VastError`` and changes nothing.
        """
        pending = [job for job in self._store.load_all() if job.state not in TERMINAL_STATES]
        if not pending:
            return []
        listed = self._vast.list_instances(self._settings.label_prefix)
        resumed: list[str] = []
        for job in pending:
            found = [item for item in listed if item.label == self._label(job.id)]
            if found:
                reattach(self._store, job, found, self._clock, self._destroy)
                resumed.append(job.id)
            else:
                mark_lost(self._store, job)
        return resumed

    def reap_orphans(self) -> list[int]:
        """Destroy labelled machines whose job is finished or unknown. Returns the ids destroyed.

        A machine belonging to an unfinished job is never touched, nor is one without the
        app's label prefix. Jobs with an unconfirmed destroy are settled first.
        """
        self.retry_unconfirmed_destroys()
        prefix = self._settings.label_prefix
        destroyed: list[int] = []
        for instance in self._vast.list_instances(prefix):
            if not instance.label or not instance.label.startswith(prefix):
                continue
            if belongs_to_unfinished_job(self._store, instance.label[len(prefix) :]):
                continue
            logger.warning("Destroying orphaned machine %s (%s)", instance.id, instance.label)
            if self._destroy(instance.id):
                destroyed.append(instance.id)
            else:
                logger.critical("Orphaned machine %s may still be running", instance.id)
        return destroyed

    def retry_unconfirmed_destroys(self) -> list[Job]:
        """Retry shutting down machines of finished jobs whose destroy was never confirmed.

        On success the job keeps its outcome, its machine fields and the warning are cleared,
        and its cost is banked. Jobs still unconfirmed are left untouched. Returns the jobs
        settled. Safe to call periodically.
        """
        settled: list[Job] = []
        for job in self._store.load_all():
            if job.state not in TERMINAL_STATES or job.instance_id is None:
                continue
            with self._running_lock:
                if job.id in self._running:
                    continue
            if not self._destroy(job.instance_id):
                continue
            updated = settled_job(job, self._cost(job))
            self._store.save(updated)
            settled.append(updated)
            logger.info("Machine %s of job %s is now confirmed gone", job.instance_id, job.id)
        return settled

    def raise_cap_and_continue(self, job_id: str, new_cap: float) -> Job:
        """Give a job stopped at its cap a higher cap; ``run`` then resumes from its checkpoints.

        If the job's machine was never confirmed gone, ``run`` goes back to it (its billing time
        still counts towards the cap).
        """
        job = self._store.get(job_id)
        if job.state is not JobState.STOPPED_AT_CAP:
            raise ValueError("Only a job stopped at its spending cap can continue.")
        spent = max(job.banked_cost, self._cost(job))
        if not math.isfinite(new_cap) or new_cap <= spent:
            raise ValueError(f"The new cap must be more than the ${spent:.2f} already spent.")
        updated = job.with_(
            state=JobState.PREPARING,
            cost_cap=new_cap,
            error=None,
            retries=0,
            message=f"Ready to continue with a spending cap of ${new_cap:.2f}",
        )
        self._store.save(updated)
        return updated

    # --- preparation ----------------------------------------------------------------------------

    def _label(self, job_id: str) -> str:
        return f"{self._settings.label_prefix}{job_id}"

    def _cost(self, job: Job) -> float:
        return machine_cost(
            job.banked_cost, job.machine_started_at, job.price_per_hour, self._clock()
        )

    def _load_secrets(self, ctx: _JobRun) -> str:
        try:
            vast_key = self._creds.get(VAST_API_KEY) or ""
            hf_token = self._creds.get(HF_TOKEN) or ""
        except CredentialError as exc:
            raise _JobFailed(f"Could not read the saved keys: {exc}") from None
        ctx.secrets = [vast_key, hf_token]
        if not hf_token:
            raise _JobFailed(MSG_NO_TOKEN)
        return hf_token

    def _ensure_prepared(self, ctx: _JobRun) -> None:
        job = ctx.job
        if job.audio_path and job.duration_s is not None and Path(job.audio_path).is_file():
            return  # prepared on an earlier run (resume or raised cap)
        if ctx.cancel.is_set():
            raise _Cancelled()
        ctx.save(state=JobState.PREPARING, message="Preparing the audio")
        try:
            prepared = self._prepare(job)
        except MediaPrepError as exc:
            raise _JobFailed(str(exc)) from None
        changes: dict[str, object] = {
            "audio_path": str(prepared.path),
            "duration_s": prepared.duration_s,
        }
        if job.is_youtube and job.name == UNTITLED:
            changes["name"] = safe_folder_name(prepared.title)
        ctx.save(**changes)

    # --- machines -------------------------------------------------------------------------------

    def _run_machines(self, ctx: _JobRun, hf_token: str) -> None:
        """Run on one machine after another until the job ends or retries run out."""
        while self._one_machine(ctx, hf_token):
            retries = ctx.job.retries + 1
            if retries > self._max_retries:
                raise _JobFailed(
                    f"The rented machine stopped working {retries} times, so the job was "
                    "stopped. Finished parts are saved."
                )
            ctx.save(
                retries=retries,
                state=JobState.STARTING_MACHINE,
                message="Renting another machine and continuing from the saved parts",
            )

    def _one_machine(self, ctx: _JobRun, hf_token: str) -> bool:
        """Run on one machine. True if it was lost (and shut down); False once the job is DONE.

        Cap, cancel and failure propagate to ``run``, which shuts the machine down.
        """
        fresh = ctx.job.instance_id is None
        if fresh:
            self._rent(ctx, self._choose_offer(ctx))
        session: RemoteSession | None = None
        try:
            session = self._start_machine(ctx, fresh)
            self._upload_and_launch(ctx, session, hf_token)
            self._poll(ctx, session)
        except _MachineLost as exc:
            logger.warning("Machine for job %s was lost: %s", ctx.job.id, exc)
            self._retire_lost_machine(ctx, str(exc))
            return True
        except _CapReached:
            if session is not None:
                self._try_pull_checkpoints(ctx, session)
            raise
        finally:
            if session is not None:
                self._close_quietly(session)
        self._settle(ctx, JobState.DONE, MSG_DONE)
        return False

    def _choose_offer(self, ctx: _JobRun) -> Offer:
        if ctx.cancel.is_set():
            raise _Cancelled()
        ctx.save(state=JobState.STARTING_MACHINE, message="Looking for a machine")
        try:
            balance = self._vast.get_balance()
            offers = rank_offers(self._vast.search_offers(self._settings), self._settings)
        except VastError as exc:
            raise _JobFailed(f"Could not check Vast.ai, so nothing was rented. {exc}") from None
        if not offers:
            raise _JobFailed(MSG_NO_OFFER)
        offer = offers[0]
        job = ctx.job
        cap = job.cost_cap
        if cap <= 0:
            cap = default_cost_cap(estimate_job(job.duration_s or 0.0, offer.dph_total))
        remaining = cap - job.banked_cost
        if balance < remaining:
            raise _JobFailed(
                f"Your Vast.ai balance (${balance:.2f}) is less than this job's spending cap "
                f"(${remaining:.2f}). Add credit or lower the cap. Nothing was rented."
            )
        if offer.dph_total * STARTUP_BUDGET_HOURS > remaining:
            if job.banked_cost > 0:
                # A machine was already paid for (e.g. a replacement is needed): the cap is what
                # ran out, so stop at the cap; the user can raise it and continue.
                raise _CapReached()
            raise _JobFailed(
                f"The cheapest suitable machine costs ${offer.dph_total:.2f} an hour, which the "
                f"spending cap (${remaining:.2f}) cannot cover even for start-up. "
                "Choose a higher cap and start the job again. Nothing was rented."
            )
        ctx.save(cost_cap=cap, price_per_hour=offer.dph_total)
        return offer

    def _rent(self, ctx: _JobRun, offer: Offer) -> None:
        label = self._label(ctx.job.id)
        # Billing starts with the create call, so the start time is saved before it.
        ctx.save(machine_started_at=self._clock(), message="Renting a machine")
        try:
            instance_id = self._vast.create_instance(
                offer.id,
                self._settings.image,
                label,
                self._settings.disk_gb,
                build_onstart(self._settings),
                {},
            )
        except VastError as exc:
            definite = not isinstance(exc, VastAmbiguousCreateError)
            if definite and exc.status_code in DEFINITE_REFUSAL_CODES:
                ctx.save(machine_started_at=None)
                raise _JobFailed(
                    f"Vast.ai would not rent the machine, so nothing was rented. {exc}"
                ) from None
            raise self._unclear_create(ctx, label) from None
        ctx.save(instance_id=instance_id, message="Starting the machine")
        logger.info("Job %s rented instance %s", ctx.job.id, instance_id)

    def _unclear_create(self, ctx: _JobRun, label: str) -> _JobFailed:
        """Find and destroy any machine a create with an unknown outcome may have rented."""
        logger.warning("Renting for job %s had an unclear outcome; looking for it", ctx.job.id)
        found = sweep_label(self._vast, label, self._clock, self._sleep)
        intro = "Renting a machine did not get a clear answer from Vast.ai."
        if found is None:
            ctx.save(machine_started_at=None)
            return _JobFailed(
                f"{intro} Your machines could not be checked, so a machine labelled {label} "
                "may be running: check the Vast.ai console."
            )
        stuck = [i for i in sorted(found) if not self._destroy(i)]
        cost = self._cost(ctx.job) if found else ctx.job.banked_cost
        ctx.save(banked_cost=cost, cost_so_far=cost, machine_started_at=None)
        if stuck:
            return _JobFailed(" ".join([intro, *(still_running_text(i) for i in stuck)]))
        if found:
            return _JobFailed(
                f"{intro} The machine it rented anyway has been shut down. Try again."
            )
        return _JobFailed(f"{intro} No machine labelled {label} appeared. Try again.")

    def _start_machine(self, ctx: _JobRun, fresh: bool) -> RemoteSession:
        instance_id = ctx.job.instance_id
        assert instance_id is not None
        if fresh:
            self._attach_key(ctx, instance_id)
        else:
            self._attach_key_once(instance_id)
        instance = self._wait_for_endpoint(ctx, instance_id)
        self._guard(ctx)
        try:
            return self._session_factory(instance)
        except LINK_ERRORS as exc:
            raise _MachineLost(f"could not connect to it ({type(exc).__name__})") from None

    def _attach_key(self, ctx: _JobRun, instance_id: int) -> None:
        for attempt in range(1, ATTACH_ATTEMPTS + 1):
            self._guard(ctx)
            try:
                self._vast.attach_ssh_key(instance_id, self._public_key)
                return
            except VastError as exc:
                if attempt == ATTACH_ATTEMPTS:
                    raise _JobFailed(MSG_KEY_REFUSED) from None
                logger.info("Could not attach the SSH key yet (%d): %s", attempt, exc)
                self._sleep(ATTACH_RETRY_S)

    def _attach_key_once(self, instance_id: int) -> None:
        """Re-attached machine: the key is normally there already, so one try is enough."""
        try:
            self._vast.attach_ssh_key(instance_id, self._public_key)
        except VastError as exc:
            logger.info("Re-attaching the SSH key to %s failed (may be fine): %s", instance_id, exc)

    def _wait_for_endpoint(self, ctx: _JobRun, instance_id: int) -> Instance:
        deadline = self._clock() + STARTUP_TIMEOUT_S
        while self._clock() < deadline:
            self._guard(ctx)
            try:
                instance = self._vast.get_instance(instance_id)
            except VastError as exc:
                logger.warning("Could not check the machine yet: %s", exc)
                self._sleep(self._poll_interval_s)
                continue
            if instance is None:
                raise _MachineLost("Vast.ai no longer lists it")
            if instance.status in DEAD_INSTANCE_STATUSES:
                raise _MachineLost(f"it went to status {instance.status}")
            if instance.status == "running" and instance.ssh_host and instance.ssh_port:
                return instance
            self._sleep(self._poll_interval_s)
        raise _MachineLost(f"it was not running after {STARTUP_TIMEOUT_S / 60:.0f} minutes")

    # --- running the job ------------------------------------------------------------------------

    def _upload_and_launch(self, ctx: _JobRun, session: RemoteSession, hf_token: str) -> None:
        remote_dir = self._settings.remote_job_dir
        try:
            if parse_status(session.read_text(f"{remote_dir}/{STATUS_FILE}")) is not None:
                logger.info("Job %s is already running on its machine", ctx.job.id)
                return
            self._upload_inputs(ctx, session)
            self._guard(ctx)
            session.launch(runner_command(remote_dir), {RUNNER_TOKEN_ENV: hf_token})
        except LINK_ERRORS as exc:
            raise _MachineLost(f"uploading or starting failed ({type(exc).__name__})") from None
        except RuntimeError:
            raise _MachineLost("the job could not be started on it") from None
        ctx.save(message="Starting the transcription")

    def _upload_inputs(self, ctx: _JobRun, session: RemoteSession) -> None:
        ctx.save(state=JobState.UPLOADING, message="Uploading the audio")
        remote_dir = self._settings.remote_job_dir
        job_dir = self._store.job_dir(ctx.job.id)

        def guard_progress(_sent: int, _total: int) -> None:
            self._guard(ctx)

        assert ctx.job.audio_path is not None
        audio = Path(ctx.job.audio_path)
        session.upload(audio, f"{remote_dir}/{JOB_INPUT_DIR}/{audio.name}", guard_progress)
        for local in local_checkpoints(job_dir / LOCAL_CHECKPOINT_DIR):
            remote = f"{remote_dir}/{JOB_CHECKPOINT_DIR}/{local.name}"
            session.upload(local, remote, guard_progress)
        spec = job_dir / REMOTE_SPEC_NAME
        write_json_atomic(spec, {"max_runtime_s": max_runtime_s(ctx.job.duration_s or 0.0)})
        session.upload(spec, f"{remote_dir}/{JOB_FILE}", guard_progress)

    def _poll(self, ctx: _JobRun, session: RemoteSession) -> None:
        """Poll until the outputs are downloaded. Raises for cap, cancel, failure, loss."""
        unreachable = 0
        while True:
            self._guard(ctx)
            ctx.save(cost_so_far=self._cost(ctx.job))
            try:
                if self._poll_once(ctx, session):
                    return
                unreachable = 0
            except LINK_ERRORS as exc:
                unreachable += 1
                logger.warning("Machine not reachable (%d in a row): %s", unreachable, exc)
                self._check_still_listed(ctx)
                if unreachable >= MAX_UNREACHABLE_POLLS:
                    raise _MachineLost(
                        f"it could not be reached {unreachable} times in a row"
                    ) from None
            self._sleep(self._poll_interval_s)

    def _poll_once(self, ctx: _JobRun, session: RemoteSession) -> bool:
        """One status read. True once the job is done and its outputs are on the PC."""
        remote_dir = self._settings.remote_job_dir
        status = parse_status(session.read_text(f"{remote_dir}/{STATUS_FILE}"))
        if status is None:
            return False
        job_dir = self._store.job_dir(ctx.job.id)
        pull_checkpoints(
            session, f"{remote_dir}/{JOB_CHECKPOINT_DIR}", job_dir / LOCAL_CHECKPOINT_DIR
        )
        if status.state == "failed":
            reason = status.error or "no reason was given"
            raise _JobFailed(f"The transcription failed on the machine: {reason}")
        if status.state == "done":
            self._collect_outputs(ctx, session)
            return True
        self._show_status(ctx, status)
        return False

    def _show_status(self, ctx: _JobRun, status: RemoteStatus) -> None:
        try:
            state = state_for_stage(status.stage)
        except ValueError:
            state = ctx.job.state
        ctx.save(
            state=state,
            percent=max(0, min(100, status.percent)),
            message=status.message,
            chunk=status.chunk if _is_int(status.chunk) else None,
            chunks=status.chunks if _is_int(status.chunks) else None,
        )

    def _collect_outputs(self, ctx: _JobRun, session: RemoteSession) -> None:
        ctx.save(state=JobState.SAVING, message="Downloading the transcript")
        out = self._store.job_dir(ctx.job.id) / LOCAL_OUTPUT_DIR
        remote_out = f"{self._settings.remote_job_dir}/{JOB_OUTPUT_DIR}"
        problems = pull_outputs(session, remote_out, out)
        if problems:
            raise _JobFailed(
                "The machine finished, but these transcript files did not arrive complete: "
                f"{', '.join(problems)}."
            )
        ctx.save(output_dir=str(out), percent=100)

    def _try_pull_checkpoints(self, ctx: _JobRun, session: RemoteSession) -> None:
        remote = f"{self._settings.remote_job_dir}/{JOB_CHECKPOINT_DIR}"
        local = self._store.job_dir(ctx.job.id) / LOCAL_CHECKPOINT_DIR
        try:
            pull_checkpoints(session, remote, local)
        except LINK_ERRORS as exc:
            logger.warning("Could not save the last finished parts: %s", type(exc).__name__)

    def _check_still_listed(self, ctx: _JobRun) -> None:
        instance_id = ctx.job.instance_id
        if instance_id is None:
            return
        try:
            if self._vast.get_instance(instance_id) is None:
                raise _MachineLost("Vast.ai no longer lists it")
        except VastError as exc:
            logger.warning("Could not check the machine: %s", exc)

    def _guard(self, ctx: _JobRun) -> None:
        """Raise if the user cancelled or the running machine has reached the spending cap."""
        if ctx.cancel.is_set():
            raise _Cancelled()
        job = ctx.job
        if job.machine_started_at is None or job.cost_cap <= 0:
            return
        if self._cost(job) >= job.cost_cap:
            raise _CapReached()

    # --- shutting machines down -----------------------------------------------------------------

    def _destroy(self, instance_id: int) -> bool:
        return destroy_and_confirm(
            self._vast, instance_id, self._clock, self._sleep, self._settings.label_prefix
        )

    def _settle(
        self, ctx: _JobRun, state: JobState, message: str, error: str | None = None
    ) -> None:
        """Shut down the job's machine (if any), bank its cost and save the final state.

        If the shutdown is not confirmed, ``instance_id`` and ``machine_started_at`` stay set
        (the machine may still be billing) and the message says so; see the module docstring.
        """
        changes: dict[str, object] = {"state": state, "message": message, "error": error}
        instance_id = ctx.job.instance_id
        if instance_id is not None and not self._destroy(instance_id):
            warning = still_running_text(instance_id)
            changes["message"] = f"{message} {warning}"
            changes["error"] = f"{error} {warning}" if error else warning
            changes["cost_so_far"] = self._cost(ctx.job)
            ctx.save(**changes)
            return
        cost = self._cost(ctx.job)
        changes.update(
            instance_id=None, banked_cost=cost, cost_so_far=cost, machine_started_at=None
        )
        ctx.save(**changes)

    def _retire_lost_machine(self, ctx: _JobRun, reason: str) -> None:
        instance_id = ctx.job.instance_id
        if instance_id is not None and not self._destroy(instance_id):
            raise _JobFailed(f"The machine stopped working ({reason}) and could not be shut down.")
        cost = self._cost(ctx.job)
        ctx.save(
            instance_id=None,
            banked_cost=cost,
            cost_so_far=cost,
            machine_started_at=None,
            message=f"The machine stopped working: {reason}",
        )

    @staticmethod
    def _close_quietly(session: RemoteSession) -> None:
        try:
            session.close()
        except LINK_ERRORS:
            logger.debug("SSH session was already closed")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
