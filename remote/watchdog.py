"""Self-destruct watchdog: a safety net that runs ON the rented Vast.ai machine.

It destroys the machine when the job has been finished for a grace period, or when an
overall runtime limit passes, so a crashed or closed desktop app can never leave the
machine billing indefinitely. Stdlib only. The container destroys itself with the
``vastai`` CLI using its own ``CONTAINER_ID`` / ``CONTAINER_API_KEY`` env vars; the API
key is never logged.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from common.jsonio import read_json
from common.protocol import GRACE_AFTER_FINISH_S, JOB_FILE, STATUS_FILE, RemoteStatus

logger = logging.getLogger(__name__)

DEFAULT_MAX_RUNTIME_S = 10800.0
POLL_INTERVAL_S = 60.0
DESTROY_TIMEOUT_S = 120.0
FINISHED_STATES = ("done", "failed")
ENV_CONTAINER_ID = "CONTAINER_ID"
ENV_API_KEY = "CONTAINER_API_KEY"

EXIT_OK = 0


def should_self_destruct(
    now: float,
    started_at: float,
    finished_at: float | None,
    grace_s: float,
    max_runtime_s: float,
) -> bool:
    """True once the post-finish grace period or the overall runtime limit has passed."""
    if finished_at is not None and now - finished_at >= grace_s:
        return True
    return now - started_at >= max_runtime_s


def read_max_runtime_s(job_dir: Path, default_s: float) -> float:
    """``max_runtime_s`` from ``job.json``; ``default_s`` if absent or not a positive number."""
    data = read_json(job_dir / JOB_FILE)
    if not isinstance(data, dict):
        return default_s
    value = data.get("max_runtime_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return default_s
    return float(value)


def read_finished_at(job_dir: Path) -> float | None:
    """``updated_at`` of a ``done``/``failed`` status; None for anything else.

    A missing, corrupt, or malformed ``status.json`` means "not finished yet", so a bad
    file can never crash the loop or trigger an early destroy.
    """
    data = read_json(job_dir / STATUS_FILE)
    if not isinstance(data, dict):
        return None
    try:
        status = RemoteStatus(**data)
    except TypeError:
        logger.warning("Ignoring malformed %s", STATUS_FILE)
        return None
    updated_at = status.updated_at
    if status.state not in FINISHED_STATES:
        return None
    if isinstance(updated_at, bool) or not isinstance(updated_at, (int, float)):
        return None
    return float(updated_at)


def destroy_self(
    run: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run,
    env: Mapping[str, str] = os.environ,
) -> None:
    """Destroy this container via the ``vastai`` CLI; raise RuntimeError on any failure."""
    container_id = env.get(ENV_CONTAINER_ID, "")
    api_key = env.get(ENV_API_KEY, "")
    if not container_id or not api_key:
        raise RuntimeError(f"{ENV_CONTAINER_ID} and {ENV_API_KEY} must be set to self-destruct")
    args = ["vastai", "destroy", "instance", container_id, "--api-key", api_key]
    try:
        run(args, check=True, capture_output=True, text=True, timeout=DESTROY_TIMEOUT_S)
    except subprocess.CalledProcessError as exc:
        detail = _scrub(str(exc.stderr or "").strip(), api_key)
        raise RuntimeError(f"vastai destroy exited {exc.returncode}: {detail}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("vastai destroy timed out") from None
    except OSError as exc:
        raise RuntimeError(f"could not run vastai: {_scrub(str(exc), api_key)}") from None


def _scrub(text: str, secret: str) -> str:
    return text.replace(secret, "***")


def main(
    argv: list[str] | None = None,
    *,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    destroy: Callable[[], None] = destroy_self,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> int:
    """Poll the job folder and destroy the machine when it is due. Returns 0 after destroy."""
    parser = argparse.ArgumentParser(description="Self-destruct watchdog for a rented machine.")
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    job_dir: Path = args.job_dir

    started_at = clock()
    logger.info("Watchdog started for %s", job_dir)
    while True:
        now = clock()
        if should_self_destruct(
            now=now,
            started_at=started_at,
            finished_at=read_finished_at(job_dir),
            grace_s=GRACE_AFTER_FINISH_S,
            max_runtime_s=read_max_runtime_s(job_dir, DEFAULT_MAX_RUNTIME_S),
        ):
            logger.warning("Self-destruct due; destroying this machine")
            try:
                destroy()
            except RuntimeError as exc:
                logger.error("Self-destruct failed, will retry: %s", exc)
            else:
                return EXIT_OK
        sleep(poll_interval_s)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
