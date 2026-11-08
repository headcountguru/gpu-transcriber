"""Qt tests for the background workers: nothing may escape a QThread."""

import threading
from pathlib import Path

from app.gui.worker import JobWorker, TaskWorker, plain_error
from app.jobs import JobState, JobStore
from app.media_prep import MediaPrepError
from app.orchestrator import JobAlreadyRunningError
from tests.gui_fakes import FakeOrchestrator, make_job

WAIT_MS = 5000


def test_job_worker_emits_updates_and_ends(qtbot: object, tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.save(make_job("w"))
    worker = JobWorker(FakeOrchestrator(store), "w")
    states: list[JobState] = []
    worker.job_updated.connect(lambda job: states.append(job.state))
    with qtbot.waitSignal(worker.ended, timeout=WAIT_MS) as blocker:  # type: ignore[attr-defined]
        worker.start()
    worker.wait()
    assert blocker.args == ["w", ""]
    qtbot.waitUntil(lambda: states[-1:] == [JobState.DONE], timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_job_worker_turns_any_exception_into_a_message(qtbot: object, tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.save(make_job("w"))
    orch = FakeOrchestrator(store)
    orch.run_error = KeyError("secret-ish detail")
    worker = JobWorker(orch, "w")
    with qtbot.waitSignal(worker.ended, timeout=WAIT_MS) as blocker:  # type: ignore[attr-defined]
        worker.start()
    worker.wait()
    job_id, message = blocker.args
    assert job_id == "w"
    assert "Something went wrong" in message
    assert "secret-ish" not in message


def test_job_worker_reports_already_running(qtbot: object, tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.save(make_job("w"))
    orch = FakeOrchestrator(store)
    orch.run_error = JobAlreadyRunningError("This job is already running.")
    worker = JobWorker(orch, "w")
    with qtbot.waitSignal(worker.ended, timeout=WAIT_MS) as blocker:  # type: ignore[attr-defined]
        worker.start()
    worker.wait()
    assert blocker.args[1] == "This job is already running."


def test_job_worker_cancel_sets_the_event(tmp_path: Path) -> None:
    worker = JobWorker(FakeOrchestrator(JobStore(tmp_path)), "w")
    assert not worker.cancel_event.is_set()
    worker.cancel()
    assert worker.cancel_event.is_set()


def test_task_worker_success_and_failure(qtbot: object) -> None:
    ok = TaskWorker(lambda: threading.get_ident())
    with qtbot.waitSignal(ok.succeeded, timeout=WAIT_MS) as blocker:  # type: ignore[attr-defined]
        ok.start()
    ok.wait()
    assert blocker.args[0] != threading.get_ident()

    def fail() -> None:
        raise MediaPrepError("Can't find that file.")

    bad = TaskWorker(fail)
    with qtbot.waitSignal(bad.failed, timeout=WAIT_MS) as blocker:  # type: ignore[attr-defined]
        bad.start()
    bad.wait()
    assert blocker.args == ["Can't find that file."]


def test_plain_error_keeps_plain_messages_and_hides_others() -> None:
    assert plain_error(MediaPrepError("Plain words.")) == "Plain words."
    hidden = plain_error(ValueError("internal detail"))
    assert "internal detail" not in hidden
    assert "ValueError" in hidden
