"""Fakes shared by the Qt tests: an orchestrator, a keyring backend and job builders.

Nothing here touches the network, Vast.ai, Windows Credential Manager or launches a program.
"""

import threading
import time
from pathlib import Path

from app.jobs import (
    Job,
    JobState,
    JobStore,
    output_file_name,  # noqa: E402
)
from common.protocol import OUTPUT_FILES

BLOCK_TIMEOUT_S = 10.0


class FakeBackend:
    """In-memory stand-in for the ``keyring`` module."""

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, name: str) -> str | None:
        return self.data.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self.data[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        del self.data[(service, name)]


def make_job(job_id: str = "job1", **changes: object) -> Job:
    base = Job(
        id=job_id,
        name=f"Recording {job_id}",
        source="https://youtu.be/abc",
        is_youtube=True,
        state=JobState.PREPARING,
        created_at=time.time(),
    )
    return base.with_(**changes)


def done_job_with_files(store: JobStore, job_id: str = "done1") -> Job:
    out = store.job_dir(job_id) / "output"
    out.mkdir(parents=True, exist_ok=True)
    job = make_job(
        job_id, state=JobState.DONE, output_dir=str(out), percent=100, cost_so_far=0.4, cost_cap=2.0
    )
    for name in OUTPUT_FILES:
        (out / output_file_name(job.name, name)).write_text("text", encoding="utf-8")
    store.save(job)
    return job


class FakeOrchestrator:
    """Records calls and the thread each one ran on; ``run`` can block until released."""

    def __init__(self, store: JobStore, final_state: JobState = JobState.DONE) -> None:
        self.store = store
        self.final_state = final_state
        self.calls: list[str] = []
        self.threads: dict[str, int] = {}
        self.run_ids: list[str] = []
        self.raised: list[tuple[str, float]] = []
        self.resume_ids: list[str] = []
        self.run_error: BaseException | None = None
        self.release = threading.Event()
        self.block = False
        self.started = threading.Event()
        self.hold_startup = False
        self.startup_gate = threading.Event()
        self._counter = 0

    def _record(self, name: str) -> None:
        self.calls.append(name)
        self.threads[name] = threading.get_ident()

    def submit(self, source: str, is_youtube: bool, name: str | None = None) -> Job:
        self._record("submit")
        self._counter += 1
        job = make_job(f"new{self._counter}", source=source, is_youtube=is_youtube)
        self.store.save(job)
        return job

    def run(self, job_id: str, on_update: object, cancel: threading.Event | None = None) -> Job:
        self._record("run")
        self.run_ids.append(job_id)
        if self.run_error is not None:
            raise self.run_error
        cancel = cancel or threading.Event()
        job = self.store.get(job_id).with_(
            state=JobState.LABELING, chunk=1, chunks=2, percent=30, message="Labeling"
        )
        self.store.save(job)
        on_update(job)  # type: ignore[operator]
        self.started.set()
        deadline = time.monotonic() + BLOCK_TIMEOUT_S
        while self.block and not self.release.is_set() and not cancel.is_set():
            if time.monotonic() > deadline:
                break
            time.sleep(0.01)
        final = JobState.CANCELLED if cancel.is_set() else self.final_state
        job = job.with_(state=final, message=f"{final.value} now")
        self.store.save(job)
        on_update(job)  # type: ignore[operator]
        return job

    def resume_pending(self) -> list[str]:
        self._record("resume_pending")
        if self.hold_startup:
            self.startup_gate.wait(BLOCK_TIMEOUT_S)
        return list(self.resume_ids)

    def reap_orphans(self) -> list[int]:
        self._record("reap_orphans")
        return []

    def retry_unconfirmed_destroys(self) -> list[Job]:
        self._record("retry_unconfirmed_destroys")
        return []

    def raise_cap_and_continue(self, job_id: str, new_cap: float) -> Job:
        self._record("raise_cap_and_continue")
        self.raised.append((job_id, new_cap))
        job = self.store.get(job_id).with_(state=JobState.PREPARING, cost_cap=new_cap)
        self.store.save(job)
        return job


def write_audio(folder: Path, name: str = "talk.mp3") -> Path:
    path = folder / name
    path.write_bytes(b"not really audio")
    return path
