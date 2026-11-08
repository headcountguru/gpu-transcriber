"""The start-up check: raise-cap waits for it, a failed check stays visible and is retried."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from app.credentials import CredentialStore
from app.gui.main_window import MainWindow
from app.jobs import JobState, JobStore
from tests.gui_fakes import FakeBackend, FakeOrchestrator, make_job

WAIT_MS = 5000
YOUTUBE = "https://www.youtube.com/watch?v=abc123"


@pytest.fixture()
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs")


@pytest.fixture()
def orch(store: JobStore) -> Iterator[FakeOrchestrator]:
    fake = FakeOrchestrator(store)
    yield fake
    fake.release.set()
    fake.startup_gate.set()


@pytest.fixture()
def make_window(
    qtbot: object, orch: FakeOrchestrator, store: JobStore
) -> Iterator[Callable[..., MainWindow]]:
    windows: list[MainWindow] = []

    def build(**kwargs: object) -> MainWindow:
        options: dict[str, object] = {
            "run_startup": True,
            "confirm_close": lambda: True,
            "retry_interval_ms": 0,
        }
        options.update(kwargs)
        window = MainWindow(orch, store, CredentialStore(backend=FakeBackend()), **options)
        qtbot.addWidget(window)  # type: ignore[attr-defined]
        window.show()
        windows.append(window)
        return window

    yield build
    orch.release.set()
    orch.startup_gate.set()
    for window in windows:
        window.stop_background_timers()
        qtbot.waitUntil(lambda w=window: not w.has_background_work(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def failing_then_working(orch: FakeOrchestrator, failures: int = 1) -> None:
    real = orch.resume_pending
    left = [failures]

    def flaky() -> list[str]:
        if left[0] > 0:
            left[0] -= 1
            orch.calls.append("resume_pending")
            raise ConnectionError("Could not reach Vast.ai. Check the internet connection.")
        return real()

    orch.resume_pending = flaky  # type: ignore[method-assign]


# --- B5 -----------------------------------------------------------------------------------------


def test_raise_cap_during_the_startup_check_waits_for_it(
    qtbot: object,
    make_window: Callable[..., MainWindow],
    store: JobStore,
    orch: FakeOrchestrator,
) -> None:
    store.save(make_job("cap", state=JobState.STOPPED_AT_CAP, cost_so_far=2.0, cost_cap=2.0))
    orch.hold_startup = True
    window = make_window(ask_new_cap=lambda job: 5.0)
    qtbot.waitUntil(lambda: "resume_pending" in orch.calls, timeout=WAIT_MS)  # type: ignore[attr-defined]

    window.select_job("cap")
    window.job_detail.raise_cap_button.click()

    assert "raise_cap_and_continue" not in orch.calls
    assert "startup" in window.job_detail.notice_label.text().lower()
    orch.startup_gate.set()
    qtbot.waitUntil(lambda: orch.run_ids == ["cap"], timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert orch.calls.index("raise_cap_and_continue") > orch.calls.index("reap_orphans")
    assert orch.raised == [("cap", 5.0)]
    qtbot.waitUntil(lambda: not window.has_running_jobs(), timeout=WAIT_MS)  # type: ignore[attr-defined]


# --- B6 -----------------------------------------------------------------------------------------


def test_failed_startup_check_shows_a_lasting_warning_and_is_retried(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    failing_then_working(orch)
    window = make_window(startup_retry_ms=50)

    qtbot.waitUntil(lambda: window.startup_warning.isVisible(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert "retrying" in window.startup_warning.text().lower()
    qtbot.waitUntil(lambda: orch.calls.count("resume_pending") >= 2, timeout=WAIT_MS)  # type: ignore[attr-defined]
    qtbot.waitUntil(lambda: not window.has_background_work(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    qtbot.waitUntil(lambda: not window.startup_warning.isVisible(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_tools_message_does_not_replace_the_startup_warning(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    failing_then_working(orch, failures=10**6)
    window = make_window(startup_retry_ms=60_000)
    qtbot.waitUntil(lambda: window.startup_warning.isVisible(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    warning = window.startup_warning.text()

    window.show_tools_problem("Could not download the YouTube helper.")
    window.statusBar().showMessage("something else")

    assert window.startup_warning.isVisible()
    assert window.startup_warning.text() == warning
    assert window.tools_label.isVisible()
    assert "YouTube helper" in window.tools_label.text()


def test_startup_retry_waits_while_a_job_is_running(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    failing_then_working(orch)
    orch.block = True
    window = make_window(startup_retry_ms=20)
    qtbot.waitUntil(lambda: window.startup_warning.isVisible(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    window.new_job_panel.source_edit.setText(YOUTUBE)
    window.new_job_panel.play_button.click()
    assert orch.started.wait(5)

    qtbot.wait(200)  # type: ignore[attr-defined]  # several retry periods while the job runs
    assert orch.calls.count("resume_pending") == 1

    orch.release.set()
    qtbot.waitUntil(lambda: orch.calls.count("resume_pending") >= 2, timeout=WAIT_MS)  # type: ignore[attr-defined]
    qtbot.waitUntil(lambda: not window.has_background_work(), timeout=WAIT_MS)  # type: ignore[attr-defined]
