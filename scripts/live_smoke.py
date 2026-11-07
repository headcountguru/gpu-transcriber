"""Live smoke test: ONE real end-to-end job on a rented Vast.ai machine, with measurements.

Run it by hand, with your own keys, to find out how long a transcription takes and what it
costs, and to confirm the assumptions the code makes about Vast.ai and SSH:

    VAST_API_KEY=... HF_TOKEN=... python scripts/live_smoke.py --audio clip.wav --max-spend 1.0

The two secrets come only from environment variables. They are never logged and never written
to the report (every report is scrubbed of them before it is saved).

The machine is ALWAYS destroyed in a ``finally`` (three tries, then confirmed gone). If it
cannot be confirmed gone, the instance id is logged loudly and the exit code is non-zero.
A start-up script also runs the on-machine watchdog, so even a crashed script cannot leave
the machine billing for ever.

The pure helpers (report, stage timing, spend cap, status parsing) are importable and tested;
the live flow is not run by the test suite.
"""

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TypeVar

import paramiko

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.remote_session import (  # noqa: E402
    LAUNCH_LOG_PATH,
    RemoteSession,
    SshRemoteSession,
    ensure_ssh_key,
)
from app.settings import IMAGE, Settings, app_data_dir  # noqa: E402
from app.vast_client import (  # noqa: E402
    Instance,
    Offer,
    VastClient,
    VastError,
    rank_offers,
)
from common.jsonio import write_json_atomic  # noqa: E402
from common.protocol import (  # noqa: E402
    JOB_FILE,
    JOB_INPUT_DIR,
    JOB_OUTPUT_DIR,
    OUTPUT_FILES,
    STATUS_FILE,
    RemoteStatus,
    max_runtime_s,
)

logger = logging.getLogger("live_smoke")

T = TypeVar("T")

EXIT_OK = 0
EXIT_JOB_PROBLEM = 1
EXIT_REFUSED = 2
EXIT_MACHINE_NOT_CONFIRMED_GONE = 3

ENV_VAST_KEY = "VAST_API_KEY"
ENV_HF_TOKEN = "HF_TOKEN"

REPORT_DIR = REPO_ROOT / "docs" / "measurements"
REMOTE_AUDIO_NAME = "audio.opus"
LOCAL_OPUS_BITRATE = "32k"
LOCAL_SAMPLE_RATE = "16000"

SECONDS_PER_HOUR = 3600.0
STARTUP_TIMEOUT_S = 1200.0
SSH_TIMEOUT_S = 300.0
POLL_INTERVAL_S = 10.0
MAX_LINK_ERRORS = 18  # consecutive failed status reads (3 minutes) before giving up
DESTROY_ATTEMPTS = 3
DESTROY_RETRY_S = 5.0
CONFIRM_GONE_TIMEOUT_S = 90.0
CONFIRM_POLL_S = 5.0
SELF_DESTROY_TIMEOUT_S = 300.0
LOG_EXCERPT_CHARS = 600
MAX_ENDPOINT_ERRORS = 3

RUNNER_PYTHONPATH = "/app:/app/remote"
SELF_DESTROY_COMMAND = "vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY"

# Assumption keys, in report order.
A_SSH = "ssh"
A_DETACH = "detach"
A_PYTHON = "python"
A_RENAME = "rename"
A_ENDPOINTS = "endpoints"
A_RELIABILITY = "reliability"
A_SELF_DESTROY = "self_destroy"
ASSUMPTION_NAMES = {
    A_SSH: "SSH connects to the custom image as the configured user",
    A_DETACH: "A runner started with `launch` survives the exec channel closing",
    A_PYTHON: "python and the image environment are available to the non-login exec shell",
    A_RENAME: "`posix_rename` works on the machine's sshd (uploads end atomically)",
    A_ENDPOINTS: "Vast search / create / list / destroy endpoints behave as coded",
    A_RELIABILITY: "The `reliability` server-side filter matches the parsed `reliability` field",
    A_SELF_DESTROY: "The machine can destroy itself with CONTAINER_ID / CONTAINER_API_KEY",
}

_LINK_ERRORS = (OSError, EOFError, paramiko.SSHException)
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


# --- pure helpers (tested) ---------------------------------------------------------------------


@dataclass(frozen=True)
class Assumption:
    """One row of the live-assumption table. ``result`` is PASS, FAIL or NOT RUN."""

    name: str
    result: str
    note: str = ""


def _clock_text(seconds: float) -> str:
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_report(
    timings: dict[str, float], offer: Offer, audio_seconds: float, total_cost: float
) -> str:
    """Markdown tables: every timed step, the total, the GPU, the price, the cost and the ratio.

    ``timings`` maps a step name (startup, upload, each pipeline stage, download, ...) to seconds.
    """
    total_s = sum(timings.values())
    ratio = f"{total_s / audio_seconds:.2f}x" if audio_seconds > 0 else "n/a"
    lines = ["## Timings", "", "| Step | Seconds | h:mm:ss |", "|---|---:|---:|"]
    for step, seconds in timings.items():
        lines.append(f"| {step} | {seconds:.1f} | {_clock_text(seconds)} |")
    lines.append(f"| **Total** | **{total_s:.1f}** | **{_clock_text(total_s)}** |")
    lines += [
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| GPU | {offer.gpu_name} |",
        f"| Price per hour | ${offer.dph_total:.2f} |",
        f"| Total cost | ${total_cost:.2f} |",
        f"| Audio length | {audio_seconds:.0f} s ({_clock_text(audio_seconds)}) |",
        f"| Total time / audio length | {ratio} |",
    ]
    return "\n".join(lines)


def scrub(text: str, secrets: list[str]) -> str:
    """Replace every secret in ``text`` with ``***``."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def build_report(
    *,
    timings: dict[str, float],
    offer: Offer,
    audio_seconds: float,
    total_cost: float,
    command: str,
    assumptions: list[Assumption],
    notes: list[str],
    secrets: list[str],
) -> str:
    """The full markdown report, scrubbed of ``secrets``."""
    parts = [
        "# Live smoke measurement",
        "",
        "Command (no secrets; keys come from the environment):",
        "",
        "```",
        command,
        "```",
        "",
        format_report(timings, offer, audio_seconds, total_cost),
        "",
        "## Live assumption checks",
        "",
        "| Assumption | Result | Notes |",
        "|---|---|---|",
    ]
    for item in assumptions:
        note = item.note.replace("|", "/").replace("\n", " ")
        parts.append(f"| {item.name} | {item.result} | {note} |")
    parts += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    return scrub("\n".join(parts) + "\n", secrets)


def stage_durations(first_seen: dict[str, float], end: float) -> dict[str, float]:
    """Seconds spent in each stage: from its first sighting to the next stage's (or ``end``)."""
    ordered = sorted(first_seen.items(), key=lambda item: item[1])
    durations: dict[str, float] = {}
    for index, (stage, seen) in enumerate(ordered):
        following = ordered[index + 1][1] if index + 1 < len(ordered) else end
        durations[stage] = max(0.0, following - seen)
    return durations


def spend_cap_reached(hours: float, price_per_hour: float, max_spend: float) -> bool:
    """True once the rental has cost ``max_spend`` or more."""
    return hours * price_per_hour >= max_spend


def parse_status(text: str | None) -> RemoteStatus | None:
    """Parse ``status.json`` text; a missing, corrupt or unexpected file means "no status yet"."""
    if not text:
        return None
    try:
        status = RemoteStatus.from_json(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(status.state, str) or not isinstance(status.stage, str):
        return None
    return status


def parse_ffmpeg_duration(stderr: str) -> float | None:
    """Seconds from the first ``Duration: HH:MM:SS.xx`` line ffmpeg prints, if any."""
    match = _DURATION_RE.search(stderr)
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def judge_reliability_filter(
    filtered: list[Offer], unfiltered: list[Offer], min_reliability: float
) -> tuple[str, str]:
    """Compare the server-side ``reliability`` filter against the parsed field."""
    if not filtered:
        return "FAIL", "the filtered search returned no offers, so nothing could be compared"
    too_low = [o.id for o in filtered if o.reliability < min_reliability]
    if too_low:
        return "FAIL", f"filtered offers below {min_reliability}: {too_low[:5]}"
    below = [o for o in unfiltered if o.reliability < min_reliability]
    note = f"{len(filtered)} filtered offers all >= {min_reliability}"
    if below:
        return "PASS", f"{note}; the unfiltered search had {len(below)} below it, none leaked"
    return "PASS", f"{note}; the unfiltered search had none below it, so exclusion was not shown"


# --- live-run state ----------------------------------------------------------------------------


class AssumptionLog:
    """The seven assumptions, each starting as NOT RUN until a live check records a result."""

    def __init__(self) -> None:
        self._items = {key: Assumption(name, "NOT RUN") for key, name in ASSUMPTION_NAMES.items()}

    def record(self, key: str, result: str, note: str = "") -> None:
        self._items[key] = Assumption(ASSUMPTION_NAMES[key], result, note)

    def items(self) -> list[Assumption]:
        return list(self._items.values())


class EndpointTracker:
    """Runs Vast calls and remembers which worked and which failed (assumption e)."""

    NEEDED = ("search", "create", "list", "destroy")

    def __init__(self) -> None:
        self._ok: dict[str, int] = {}
        self._errors: dict[str, list[str]] = {}

    def call(self, name: str, func: Callable[[], T]) -> T:
        try:
            value = func()
        except VastError as exc:
            self.fail(name, f"HTTP {exc.status_code}: {exc}" if exc.status_code else str(exc))
            raise
        self._ok[name] = self._ok.get(name, 0) + 1
        return value

    def fail(self, name: str, message: str) -> None:
        errors = self._errors.setdefault(name, [])
        if message not in errors and len(errors) < MAX_ENDPOINT_ERRORS:
            errors.append(message)

    def summary(self) -> tuple[str, str]:
        if self._errors:
            text = "; ".join(f"{name}: {' / '.join(msgs)}" for name, msgs in self._errors.items())
            return "FAIL", text
        missing = [name for name in self.NEEDED if name not in self._ok]
        if missing:
            return "NOT RUN", f"not exercised: {', '.join(missing)}"
        counts = ", ".join(f"{name} x{count}" for name, count in self._ok.items())
        return "PASS", f"all returned the expected shape ({counts})"


@dataclass
class SmokeRun:
    """Mutable state of one live run, shared by the steps and the cleanup."""

    client: VastClient
    settings: Settings
    secrets: list[str]
    log: AssumptionLog = field(default_factory=AssumptionLog)
    tracker: EndpointTracker = field(default_factory=EndpointTracker)
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    label: str = ""
    instance_id: int | None = None
    created_at: float | None = None
    ended_at: float | None = None
    create_attempted: bool = False

    def billed_hours(self) -> float:
        if self.created_at is None:
            return 0.0
        return (time.monotonic() - self.created_at) / SECONDS_PER_HOUR


@dataclass(frozen=True)
class Prepared:
    """What the pre-flight steps produced."""

    opus_path: Path
    audio_seconds: float
    offer: Offer
    key_path: Path


class SmokeRefusal(RuntimeError):
    """The run cannot start (no offer fits, balance too low, bad audio). Nothing was rented."""


@dataclass(frozen=True)
class PollResult:
    """How the status polling ended. Times are seconds since polling started."""

    outcome: str  # done | failed | spend-cap | timeout | lost-connection
    first_seen: dict[str, float]
    elapsed_s: float
    status: RemoteStatus | None


def poll_job(
    session: RemoteSession,
    status_path: str,
    *,
    billed_hours: Callable[[], float],
    price_per_hour: float,
    max_spend: float,
    timeout_s: float,
    poll_interval_s: float = POLL_INTERVAL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PollResult:
    """Poll ``status.json`` until the job ends, the spend cap is hit, or time runs out."""
    start = clock()
    first_seen: dict[str, float] = {}
    last: RemoteStatus | None = None
    link_errors = 0
    while True:
        elapsed = clock() - start
        text: str | None = None
        try:
            text = session.read_text(status_path)
            link_errors = 0
        except _LINK_ERRORS as exc:
            link_errors += 1
            logger.warning("Could not read status (%d in a row): %s", link_errors, exc)
            if link_errors >= MAX_LINK_ERRORS:
                return PollResult("lost-connection", first_seen, elapsed, last)
        status = parse_status(text)
        if status is not None:
            last = status
            if status.state == "done":
                return PollResult("done", first_seen, elapsed, status)
            if status.stage not in first_seen:
                first_seen[status.stage] = elapsed
                logger.info("Stage %s at %.0fs (%d%%)", status.stage, elapsed, status.percent)
            if status.state == "failed":
                return PollResult("failed", first_seen, elapsed, status)
        if spend_cap_reached(billed_hours(), price_per_hour, max_spend):
            return PollResult("spend-cap", first_seen, elapsed, last)
        if elapsed >= timeout_s:
            return PollResult("timeout", first_seen, elapsed, last)
        sleep(poll_interval_s)


# --- live steps --------------------------------------------------------------------------------


def compress_audio(source: Path, target: Path) -> float:
    """Convert to 16 kHz mono Opus 32 kbps with the bundled ffmpeg; return the audio seconds."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", LOCAL_SAMPLE_RATE,
        "-c:a", "libopus", "-b:a", LOCAL_OPUS_BITRATE, str(target),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not target.is_file():
        raise SmokeRefusal(f"ffmpeg could not convert the audio: {result.stderr[-300:]}")
    duration = parse_ffmpeg_duration(result.stderr)
    if duration is None:
        raise SmokeRefusal("ffmpeg did not report the audio length")
    return duration


def choose_offer(run: SmokeRun, offer_id: int | None, max_spend: float) -> Offer:
    """Pick the offer to rent (cheapest fitting, or ``offer_id``); refuse if none or too poor."""
    balance = run.tracker.call("balance", run.client.get_balance)
    if balance < max_spend:
        raise SmokeRefusal(f"Vast.ai balance ${balance:.2f} is below --max-spend ${max_spend:.2f}")
    filtered = run.tracker.call("search", lambda: run.client.search_offers(run.settings))
    unfiltered = run.tracker.call("search", run.client.search_offers)
    result, note = judge_reliability_filter(filtered, unfiltered, run.settings.min_reliability)
    run.log.record(A_RELIABILITY, result, note)
    if offer_id is not None:
        offer = next((o for o in unfiltered if o.id == offer_id), None)
        if offer is None:
            raise SmokeRefusal(f"Offer {offer_id} is not in the current rentable list")
        return offer
    ranked = rank_offers(filtered, run.settings)
    if not ranked:
        raise SmokeRefusal("No machine fits the settings right now; try again later")
    return ranked[0]


def prepare(run: SmokeRun, args: argparse.Namespace, workdir: Path) -> Prepared:
    """Compress the audio, set up the SSH key, and pick the machine. Rents nothing."""
    started = time.monotonic()
    opus_path = workdir / REMOTE_AUDIO_NAME
    audio_seconds = compress_audio(args.audio, opus_path)
    run.timings["compress"] = time.monotonic() - started
    key_path, public_key = ensure_ssh_key(app_data_dir())
    offer = choose_offer(run, args.offer_id, args.max_spend)
    run.tracker.call("register_ssh_key", lambda: run.client.register_ssh_key(public_key))
    run.notes.append(
        f"Chosen offer {offer.id}: {offer.gpu_name}, {offer.gpu_ram_gb:.0f} GB, "
        f"datacenter flag {offer.datacenter}, reliability {offer.reliability:.3f}"
    )
    return Prepared(opus_path, audio_seconds, offer, key_path)


def onstart_script(job_dir: str) -> str:
    """Start-up script: run the watchdog so a crashed app cannot leave the machine billing."""
    return (
        f"mkdir -p {shlex.quote(job_dir)}\n"
        f"nohup env PYTHONPATH={RUNNER_PYTHONPATH} python /app/remote/watchdog.py "
        f"--job-dir {shlex.quote(job_dir)} > /tmp/watchdog.log 2>&1 &\n"
    )


def rent_machine(run: SmokeRun, offer: Offer) -> int:
    """Rent the offer and check the new instance shows up in the label listing."""
    run.label = f"{run.settings.label_prefix}smoke-{time.strftime('%Y%m%d%H%M%S')}"
    run.create_attempted = True
    run.created_at = time.monotonic()
    instance_id = run.tracker.call(
        "create",
        lambda: run.client.create_instance(
            offer.id,
            run.settings.image,
            run.label,
            run.settings.disk_gb,
            onstart_script(run.settings.remote_job_dir),
            {},
        ),
    )
    run.instance_id = instance_id
    logger.info("Rented instance %s with label %s", instance_id, run.label)
    listed = run.tracker.call("list", lambda: run.client.list_instances(run.label))
    if instance_id not in {item.id for item in listed}:
        run.tracker.fail("list", "the new instance was not in the label listing")
    return instance_id


def wait_for_endpoint(run: SmokeRun, instance_id: int) -> Instance:
    """Wait until the instance is running and reports an SSH host and port."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            instance = run.tracker.call(
                "get_instance", lambda: run.client.get_instance(instance_id)
            )
        except VastError as exc:
            logger.warning("Could not check the instance yet: %s", exc)
            time.sleep(POLL_INTERVAL_S)
            continue
        if instance is None:
            raise RuntimeError("Vast.ai no longer lists the instance")
        if instance.status in ("exited", "offline", "unknown"):
            raise RuntimeError(f"The instance went to status {instance.status}")
        if instance.status == "running" and instance.ssh_host and instance.ssh_port:
            return instance
        logger.info("Instance status: %s", instance.status)
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"The instance was not running after {STARTUP_TIMEOUT_S:.0f}s")


def connect_ssh(run: SmokeRun, instance: Instance, key_path: Path) -> SshRemoteSession:
    assert instance.ssh_host is not None and instance.ssh_port is not None
    try:
        session = SshRemoteSession.connect(
            instance.ssh_host, instance.ssh_port, str(key_path), timeout_s=SSH_TIMEOUT_S
        )
    except ConnectionError as exc:
        run.log.record(A_SSH, "FAIL", str(exc))
        raise
    run.log.record(A_SSH, "PASS", f"connected as root to {instance.ssh_host}:{instance.ssh_port}")
    return session


def probe_posix_rename(run: SmokeRun, session: SshRemoteSession, workdir: Path) -> None:
    """Rename a file over an existing one with ``posix_rename``, directly on the SFTP channel."""
    probe_dir = run.settings.remote_job_dir + "-probe"
    local = workdir / "probe.txt"
    local.write_text("x", encoding="ascii")
    sftp = session._sftp
    try:
        session.upload(local, f"{probe_dir}/a.txt")
        with sftp.open(f"{probe_dir}/b.part", "wb") as handle:
            handle.write(b"y")
        sftp.posix_rename(f"{probe_dir}/b.part", f"{probe_dir}/a.txt")
        size = sftp.stat(f"{probe_dir}/a.txt").st_size
    except (OSError, EOFError, paramiko.SSHException) as exc:
        run.log.record(A_RENAME, "FAIL", f"{type(exc).__name__}: {exc}")
        return
    run.log.record(A_RENAME, "PASS", f"renamed over an existing file (size now {size})")


def upload_inputs(run: SmokeRun, session: SshRemoteSession, prepared: Prepared, tmp: Path) -> None:
    job_dir = run.settings.remote_job_dir
    started = time.monotonic()
    session.upload(prepared.opus_path, f"{job_dir}/{JOB_INPUT_DIR}/{REMOTE_AUDIO_NAME}")
    run.timings["upload"] = time.monotonic() - started
    job_file = tmp / JOB_FILE
    write_json_atomic(job_file, {"max_runtime_s": max_runtime_s(prepared.audio_seconds)})
    session.upload(job_file, f"{job_dir}/{JOB_FILE}")


def record_runner_outcome(run: SmokeRun, session: SshRemoteSession, result: PollResult) -> None:
    """Record assumptions b and c from whether the launched runner ever wrote a status."""
    if result.first_seen or result.status is not None:
        run.log.record(A_DETACH, "PASS", "status.json kept updating after launch returned")
        run.log.record(A_PYTHON, "PASS", "`python /app/remote/run_job.py` started and wrote status")
        return
    try:
        tail = (session.read_text(LAUNCH_LOG_PATH) or "")[-LOG_EXCERPT_CHARS:]
    except _LINK_ERRORS:
        tail = "(runner log unreadable)"
    run.log.record(A_DETACH, "FAIL", "no status.json appeared after launch")
    missing_env = re.search(r"not found|No module|No such file", tail) is not None
    run.log.record(A_PYTHON, "FAIL" if missing_env else "NOT RUN", f"runner log tail: {tail}")


def download_outputs(run: SmokeRun, session: SshRemoteSession, out_dir: Path) -> bool:
    """Download the three output files; True only if all arrived."""
    started = time.monotonic()
    complete = True
    for name in OUTPUT_FILES:
        remote = f"{run.settings.remote_job_dir}/{JOB_OUTPUT_DIR}/{name}"
        try:
            session.download(remote, out_dir / name)
        except (OSError, EOFError, paramiko.SSHException) as exc:
            logger.error("Could not download %s: %s", name, exc)
            run.notes.append(f"Output file {name} was not downloaded ({type(exc).__name__})")
            complete = False
    run.timings["download"] = time.monotonic() - started
    return complete


def verify_self_destroy(run: SmokeRun, session: SshRemoteSession, instance_id: int) -> None:
    """Have the machine destroy itself, then watch Vast.ai until it is gone (assumption g)."""
    try:
        session.launch(SELF_DESTROY_COMMAND, {})
    except (RuntimeError, TimeoutError, *_LINK_ERRORS) as exc:
        run.log.record(A_SELF_DESTROY, "FAIL", f"could not launch the command: {exc}")
        return
    started = time.monotonic()
    while time.monotonic() - started < SELF_DESTROY_TIMEOUT_S:
        try:
            instance = run.client.get_instance(instance_id)
        except VastError as exc:
            logger.warning("Could not check the instance: %s", exc)
            time.sleep(POLL_INTERVAL_S)
            continue
        if instance is None:
            elapsed = time.monotonic() - started
            run.ended_at = time.monotonic()
            run.log.record(
                A_SELF_DESTROY, "PASS", f"instance gone {elapsed:.0f}s after the command"
            )
            return
        time.sleep(POLL_INTERVAL_S)
    run.log.record(
        A_SELF_DESTROY, "FAIL", f"instance still listed {SELF_DESTROY_TIMEOUT_S:.0f}s later"
    )


def run_remote_job(
    run: SmokeRun,
    args: argparse.Namespace,
    hf_token: str,
    prepared: Prepared,
    session: SshRemoteSession,
    tmp: Path,
) -> str:
    """Upload, run the pipeline, poll, and download. Returns the outcome word."""
    job_dir = run.settings.remote_job_dir
    probe_posix_rename(run, session, tmp)
    upload_inputs(run, session, prepared, tmp)
    command = (
        f"env PYTHONPATH={RUNNER_PYTHONPATH} python /app/remote/run_job.py --job-dir {job_dir}"
    )
    session.launch(command, {ENV_HF_TOKEN: hf_token})
    result = poll_job(
        session,
        f"{job_dir}/{STATUS_FILE}",
        billed_hours=run.billed_hours,
        price_per_hour=prepared.offer.dph_total,
        max_spend=args.max_spend,
        timeout_s=max_runtime_s(prepared.audio_seconds),
    )
    record_runner_outcome(run, session, result)
    if result.first_seen:
        run.timings["runner_start"] = min(result.first_seen.values())
    for stage, seconds in stage_durations(result.first_seen, result.elapsed_s).items():
        run.timings[stage] = seconds
    if result.status is not None and result.status.error:
        run.notes.append(f"Runner reported: {result.status.error}")
    if result.outcome != "done":
        return result.outcome
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not download_outputs(run, session, out_dir):
        return "download-incomplete"
    if args.verify_self_destroy and run.instance_id is not None:
        verify_self_destroy(run, session, run.instance_id)
    return "done"


def execute(
    run: SmokeRun, args: argparse.Namespace, hf_token: str, prepared: Prepared, tmp: Path
) -> str:
    """Rent, wait, run the job. Cleanup is the caller's ``finally``."""
    instance_id = rent_machine(run, prepared.offer)
    instance = wait_for_endpoint(run, instance_id)
    session = connect_ssh(run, instance, prepared.key_path)
    run.timings["startup"] = time.monotonic() - (run.created_at or time.monotonic())
    try:
        return run_remote_job(run, args, hf_token, prepared, session, tmp)
    finally:
        try:
            session.close()
        except _LINK_ERRORS:
            logger.debug("SSH session was already closed")


# --- cleanup -----------------------------------------------------------------------------------


def destroy_and_confirm(run: SmokeRun, instance_id: int) -> bool:
    """Destroy (three tries) and confirm the instance is gone. False if it may still bill."""
    try:
        if run.client.get_instance(instance_id) is None:
            logger.info("Instance %s is already gone", instance_id)
            return True
    except VastError as exc:
        logger.warning("Could not check instance %s before destroying: %s", instance_id, exc)
    for attempt in range(1, DESTROY_ATTEMPTS + 1):
        try:
            run.tracker.call("destroy", lambda: run.client.destroy_instance(instance_id))
            break
        except VastError as exc:
            logger.warning("Destroy attempt %d for %s failed: %s", attempt, instance_id, exc)
            time.sleep(DESTROY_RETRY_S)
    deadline = time.monotonic() + CONFIRM_GONE_TIMEOUT_S
    while True:
        try:
            if run.client.get_instance(instance_id) is None:
                logger.info("Confirmed instance %s is gone", instance_id)
                return True
        except VastError as exc:
            logger.warning("Could not confirm instance %s is gone: %s", instance_id, exc)
        if time.monotonic() >= deadline:
            return False
        time.sleep(CONFIRM_POLL_S)


def cleanup(run: SmokeRun) -> bool:
    """Destroy the known instance and anything carrying this run's label. True if all gone."""
    ids: set[int] = set()
    sweep_ok = True
    if run.instance_id is not None:
        ids.add(run.instance_id)
    if run.label:
        try:
            ids.update(
                i.id for i in run.tracker.call("list", lambda: run.client.list_instances(run.label))
            )
        except VastError as exc:
            sweep_ok = False
            logger.error("Could not sweep for instances labelled %s: %s", run.label, exc)
    all_gone = sweep_ok or run.instance_id is not None
    if not sweep_ok and run.instance_id is None and run.create_attempted:
        logger.critical(
            "A machine may have been rented and could not be looked for. CHECK VAST.AI."
        )
        all_gone = False
    for instance_id in sorted(ids):
        if not destroy_and_confirm(run, instance_id):
            all_gone = False
            logger.critical(
                "INSTANCE %s COULD NOT BE CONFIRMED DESTROYED. IT MAY STILL BE BILLING. "
                "Destroy it in the Vast.ai console now.",
                instance_id,
            )
    if run.ended_at is None:
        run.ended_at = time.monotonic()
    return all_gone


# --- orchestration -----------------------------------------------------------------------------


def command_line(args: argparse.Namespace) -> str:
    parts = ["python", "scripts/live_smoke.py", "--audio", str(args.audio)]
    parts += ["--max-spend", f"{args.max_spend:g}"]
    if args.offer_id is not None:
        parts += ["--offer-id", str(args.offer_id)]
    if args.verify_self_destroy:
        parts.append("--verify-self-destroy")
    return " ".join(shlex.quote(part) for part in parts)


def write_report(run: SmokeRun, args: argparse.Namespace, prepared: Prepared, outcome: str) -> Path:
    if run.created_at is not None and run.ended_at is not None:
        hours = (run.ended_at - run.created_at) / SECONDS_PER_HOUR
    else:
        hours = 0.0
    cost = hours * prepared.offer.dph_total
    run.notes.insert(0, f"Outcome: {outcome}")
    run.notes.append(
        "Stage times come from polling every "
        f"{POLL_INTERVAL_S:.0f}s, so each is accurate to about that much."
    )
    run.notes.append(
        f"Cost is rental hours x the offer's hourly price (${prepared.offer.dph_total:.2f}); "
        "it ignores any storage or bandwidth fees."
    )
    status, note = run.tracker.summary()
    run.log.record(A_ENDPOINTS, status, note)
    report = build_report(
        timings=run.timings,
        offer=prepared.offer,
        audio_seconds=prepared.audio_seconds,
        total_cost=cost,
        command=command_line(args),
        assumptions=run.log.items(),
        notes=run.notes,
        secrets=run.secrets,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")
    logger.info("Wrote report %s (cost about $%.2f)", args.report_path, cost)
    return args.report_path


def run_smoke(args: argparse.Namespace, vast_key: str, hf_token: str) -> int:
    """One real job. Always destroys the machine. Returns the process exit code."""
    run = SmokeRun(VastClient(vast_key), Settings(image=IMAGE), [vast_key, hf_token])
    with tempfile.TemporaryDirectory(prefix="gputx-smoke-") as tmp_name:
        tmp = Path(tmp_name)
        try:
            prepared = prepare(run, args, tmp)
        except (SmokeRefusal, VastError) as exc:
            logger.error("Not starting: %s", scrub(str(exc), run.secrets))
            return EXIT_REFUSED
        outcome = "error"
        try:
            outcome = execute(run, args, hf_token, prepared, tmp)
        except Exception as exc:
            logger.exception("The run failed")
            outcome = f"error: {type(exc).__name__}"
        finally:
            all_gone = cleanup(run)
            try:
                write_report(run, args, prepared, outcome)
            except OSError:
                logger.exception("Could not write the report")
    if not all_gone:
        return EXIT_MACHINE_NOT_CONFIRMED_GONE
    return EXIT_OK if outcome == "done" else EXIT_JOB_PROBLEM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ONE real transcription job on a rented Vast.ai GPU and write a "
        f"measurement report. Needs {ENV_VAST_KEY} and {ENV_HF_TOKEN} in the environment."
    )
    parser.add_argument("--audio", type=Path, required=True, help="audio or video file to process")
    parser.add_argument(
        "--max-spend", type=float, default=2.0, help="stop and destroy at this many dollars"
    )
    parser.add_argument("--offer-id", type=int, help="rent this offer instead of the cheapest fit")
    parser.add_argument(
        "--verify-self-destroy",
        action="store_true",
        help="after the job, have the machine destroy itself and watch for it to vanish",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    vast_key = os.environ.get(ENV_VAST_KEY, "")
    hf_token = os.environ.get(ENV_HF_TOKEN, "")
    if not vast_key or not hf_token:
        logger.error("Set %s and %s in the environment first.", ENV_VAST_KEY, ENV_HF_TOKEN)
        return EXIT_REFUSED
    if not args.audio.is_file():
        logger.error("Audio file not found: %s", args.audio)
        return EXIT_REFUSED
    base = f"{date.today().isoformat()}-{re.sub(r'[^A-Za-z0-9._-]+', '-', args.audio.stem)}"
    args.out_dir = REPORT_DIR / base
    args.report_path = REPORT_DIR / f"{base}.md"
    return run_smoke(args, vast_key, hf_token)


if __name__ == "__main__":
    sys.exit(main())
