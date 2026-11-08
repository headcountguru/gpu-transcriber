"""Qt tests for the main window, the new-job panel and the job detail view (offscreen)."""

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from app.credentials import CredentialStore
from app.estimate import estimate_job, format_estimate
from app.gui.job_detail import JobDetail
from app.gui.main_window import MainWindow
from app.gui.new_job_panel import ESTIMATE_PRICE_PER_HOUR, NewJobPanel
from app.gui.setup_dialog import SetupDialog
from app.hf_check import DIARIZATION_MODEL, SEGMENTATION_MODEL, HfAccess
from app.jobs import Job, JobState, JobStore
from app.media_prep import MediaPrepError
from app.orchestrator import JobAlreadyRunningError
from tests.gui_fakes import (
    FakeBackend,
    FakeOrchestrator,
    done_job_with_files,
    make_job,
    write_audio,
)

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


@pytest.fixture()
def opened() -> list[str]:
    return []


@pytest.fixture()
def make_window(
    qtbot: object, orch: FakeOrchestrator, store: JobStore, opened: list[str]
) -> Iterator[Callable[..., MainWindow]]:
    windows: list[MainWindow] = []

    def build(**kwargs: object) -> MainWindow:
        options: dict[str, object] = {
            "opener": opened.append,
            "run_startup": False,
            "confirm_close": lambda: True,
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
        qtbot.waitUntil(lambda w=window: not w.has_background_work(), timeout=WAIT_MS)  # type: ignore[attr-defined]


# --- new job panel ------------------------------------------------------------------------------


def test_play_disabled_until_input_is_valid(qtbot: object, tmp_path: Path) -> None:
    panel = NewJobPanel(probe=None, debounce_ms=0)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    assert not panel.play_button.isEnabled()
    assert panel.play_button.toolTip() == "Paste a YouTube link or choose a file"

    panel.source_edit.setText(YOUTUBE)
    assert panel.play_button.isEnabled()

    panel.source_edit.setText(str(write_audio(tmp_path)))
    assert not panel.play_button.isEnabled()  # waits for the worker's file check
    qtbot.waitUntil(panel.play_button.isEnabled, timeout=WAIT_MS)  # type: ignore[attr-defined]

    panel.source_edit.setText("")
    assert not panel.play_button.isEnabled()
    qtbot.waitUntil(lambda: not panel.is_probing(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_unc_path_is_checked_off_the_ui_thread_and_play_waits(qtbot: object) -> None:
    calls: list[tuple[str, int]] = []
    release = threading.Event()

    def slow_checker(path: str) -> bool:
        calls.append((path, threading.get_ident()))
        release.wait(5)  # a sleeping network drive
        return True

    unc = r"\\server\share\meeting.mp3"
    panel = NewJobPanel(probe=None, debounce_ms=0, file_checker=slow_checker)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.source_edit.setText(unc)
    assert not panel.play_button.isEnabled()
    assert panel.estimate_label.text() == "Checking the file…"
    qtbot.waitUntil(lambda: bool(calls), timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert not panel.play_button.isEnabled()
    release.set()
    qtbot.waitUntil(panel.play_button.isEnabled, timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert all(thread != threading.get_ident() for _path, thread in calls)
    assert calls[0][0] == unc
    qtbot.waitUntil(lambda: not panel.is_probing(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_missing_file_is_reported_by_the_worker(qtbot: object, tmp_path: Path) -> None:
    panel = NewJobPanel(probe=None, debounce_ms=0)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.source_edit.setText(str(tmp_path / "nope.mp3"))
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: "find that file" in panel.error_label.text(), timeout=WAIT_MS
    )
    assert not panel.play_button.isEnabled()
    qtbot.waitUntil(lambda: not panel.is_probing(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_typing_and_dragging_never_touch_the_filesystem_on_the_ui_thread(
    qtbot: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui_thread = threading.get_ident()
    ui_calls: list[object] = []

    def guarded(path: object) -> bool:
        if threading.get_ident() == ui_thread:
            ui_calls.append(path)
        return False  # never reach for the (fake) network share

    monkeypatch.setattr("os.path.isfile", guarded)
    monkeypatch.setattr("os.path.exists", guarded)
    panel = NewJobPanel(probe=None, debounce_ms=0)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.source_edit.setText(r"\\server\share\a.mp3")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(r"\\server\share\b.mp3")])
    enter = QDragEnterEvent(
        QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )  # fmt: skip
    panel.dragEnterEvent(enter)
    assert enter.isAccepted()
    qtbot.waitUntil(lambda: not panel.is_probing(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert ui_calls == []


def test_playlist_link_keeps_play_disabled_with_plain_error(qtbot: object) -> None:
    panel = NewJobPanel(probe=None)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.source_edit.setText("https://www.youtube.com/playlist?list=PL1")
    assert not panel.play_button.isEnabled()
    assert "single video" in panel.error_label.text()
    assert not panel.error_label.isHidden()
    assert panel.error_label.textFormat() == Qt.TextFormat.PlainText


def test_play_emits_source(qtbot: object) -> None:
    panel = NewJobPanel(probe=None)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.source_edit.setText(YOUTUBE)
    with qtbot.waitSignal(panel.play_requested, timeout=WAIT_MS) as blocker:  # type: ignore[attr-defined]
        panel.play_button.click()
    assert blocker.args == [YOUTUBE, True]


def test_drag_and_drop_file_sets_source(qtbot: object, tmp_path: Path) -> None:
    panel = NewJobPanel(probe=None, debounce_ms=0)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    audio = write_audio(tmp_path, "dropped.wav")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(audio))])

    enter = QDragEnterEvent(
        QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )  # fmt: skip
    panel.dragEnterEvent(enter)
    assert enter.isAccepted()

    drop = QDropEvent(
        QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )  # fmt: skip
    panel.dropEvent(drop)
    assert Path(panel.source_edit.text()) == audio
    qtbot.waitUntil(panel.play_button.isEnabled, timeout=WAIT_MS)  # type: ignore[attr-defined]
    qtbot.waitUntil(lambda: not panel.is_probing(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_drag_of_a_web_url_is_refused(qtbot: object) -> None:
    panel = NewJobPanel(probe=None)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    mime = QMimeData()
    mime.setUrls([QUrl("https://example.com/a.mp3")])
    enter = QDragEnterEvent(
        QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )  # fmt: skip
    enter.ignore()
    panel.dragEnterEvent(enter)
    assert not enter.isAccepted()


def test_estimate_line_updates_after_probe_off_the_ui_thread(qtbot: object) -> None:
    seen: list[tuple[str, bool, int]] = []

    def probe(source: str, is_youtube: bool) -> float:
        seen.append((source, is_youtube, threading.get_ident()))
        return 3600.0

    panel = NewJobPanel(probe=probe, debounce_ms=0)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.source_edit.setText(YOUTUBE)
    expected = format_estimate(estimate_job(3600.0, ESTIMATE_PRICE_PER_HOUR))
    qtbot.waitUntil(lambda: panel.estimate_label.text() == expected, timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert expected.startswith("Est. ")
    assert "estimate only" in panel.estimate_note.text().lower()
    assert not panel.estimate_note.isHidden()
    assert seen[0][:2] == (YOUTUBE, True)
    assert seen[0][2] != threading.get_ident()
    qtbot.waitUntil(lambda: not panel.is_probing(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_estimate_probe_error_is_shown_in_plain_words(qtbot: object) -> None:
    def probe(source: str, is_youtube: bool) -> float:
        raise MediaPrepError("That video is private or was removed.")

    panel = NewJobPanel(probe=probe, debounce_ms=0)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.source_edit.setText(YOUTUBE)
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: "private or was removed" in panel.error_label.text(), timeout=WAIT_MS
    )
    qtbot.waitUntil(lambda: not panel.is_probing(), timeout=WAIT_MS)  # type: ignore[attr-defined]


# --- main window: list, detail, play ------------------------------------------------------------


def test_empty_state_when_no_job_is_selected(make_window: Callable[..., MainWindow]) -> None:
    window = make_window()
    assert "Paste a YouTube link or choose an audio file, then press Play." in (
        window.job_detail.empty_label.text()
    )
    assert window.job_list.count() == 0


def test_job_list_shows_saved_jobs_and_selecting_shows_stage_rows(
    make_window: Callable[..., MainWindow], store: JobStore
) -> None:
    store.save(make_job("a", name="Board meeting", state=JobState.LABELING, chunk=7, chunks=12))
    done_job_with_files(store, "b")
    window = make_window()
    assert window.job_list.count() == 2
    assert "Board meeting" in window.job_item("a").name_label.text()

    window.select_job("a")
    detail = window.job_detail
    assert detail.title_label.text() == "Board meeting"
    texts = detail.stage_texts()
    assert "Labeling speakers (chunk 7 of 12)" in texts
    assert detail.stage_status("Labeling speakers (chunk 7 of 12)") == "active"
    assert detail.stage_status("Saving results") == "pending"


def test_play_submits_and_runs_the_job(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    window = make_window()
    window.new_job_panel.source_edit.setText(YOUTUBE)
    window.new_job_panel.play_button.click()
    assert orch.calls[0] == "submit"
    assert window.job_list.count() == 1
    qtbot.waitUntil(lambda: not window.has_running_jobs(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert orch.threads["run"] != threading.get_ident()
    assert window.job_detail.status_label.text() == "Done"


def test_finished_job_shows_three_open_buttons(
    make_window: Callable[..., MainWindow], store: JobStore, opened: list[str]
) -> None:
    job = done_job_with_files(store)
    window = make_window()
    window.select_job(job.id)
    detail = window.job_detail
    for button in (
        detail.open_transcript_button,
        detail.open_timestamped_button,
        detail.open_subtitles_button,
        detail.show_folder_button,
    ):
        assert button.isVisible()
    assert not detail.raise_cap_button.isVisible()
    assert not detail.cancel_button.isVisible()

    detail.open_transcript_button.click()
    detail.open_subtitles_button.click()
    detail.show_folder_button.click()
    out = Path(job.output_dir or "")
    assert opened == [str(out / "transcript.txt"), str(out / "transcript.srt"), str(out)]


def test_open_missing_file_shows_plain_message(
    make_window: Callable[..., MainWindow], store: JobStore, opened: list[str]
) -> None:
    job = done_job_with_files(store)
    (Path(job.output_dir or "") / "transcript_timestamped.txt").unlink()
    window = make_window()
    window.select_job(job.id)
    window.job_detail.open_timestamped_button.click()
    assert opened == []
    assert "can't find" in window.job_detail.notice_label.text().lower()


def test_stopped_at_cap_shows_raise_cap_button(
    make_window: Callable[..., MainWindow], store: JobStore
) -> None:
    store.save(make_job("cap", state=JobState.STOPPED_AT_CAP, cost_so_far=2.0, cost_cap=2.0))
    window = make_window()
    window.select_job("cap")
    detail = window.job_detail
    assert detail.raise_cap_button.isVisible()
    assert not detail.open_transcript_button.isVisible()
    assert detail.cost_label.text().endswith("$2.00 of $2.00 cap")


def test_raise_cap_flow_raises_and_restarts(
    qtbot: object, make_window: Callable[..., MainWindow], store: JobStore, orch: FakeOrchestrator
) -> None:
    store.save(make_job("cap", state=JobState.STOPPED_AT_CAP, cost_so_far=2.0, cost_cap=2.0))
    asked: list[Job] = []

    def ask(job: Job) -> float | None:
        asked.append(job)
        return 5.0

    window = make_window(ask_new_cap=ask)
    window.select_job("cap")
    window.job_detail.raise_cap_button.click()
    assert asked[0].id == "cap"
    assert orch.raised == [("cap", 5.0)]
    qtbot.waitUntil(lambda: orch.run_ids == ["cap"], timeout=WAIT_MS)  # type: ignore[attr-defined]
    qtbot.waitUntil(lambda: not window.has_running_jobs(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_raise_cap_cancelled_does_nothing(
    make_window: Callable[..., MainWindow], store: JobStore, orch: FakeOrchestrator
) -> None:
    store.save(make_job("cap", state=JobState.STOPPED_AT_CAP, cost_so_far=2.0, cost_cap=2.0))
    window = make_window(ask_new_cap=lambda job: None)
    window.select_job("cap")
    window.job_detail.raise_cap_button.click()
    assert orch.raised == []
    assert orch.run_ids == []


def test_raise_cap_refused_shows_plain_message(
    make_window: Callable[..., MainWindow], store: JobStore, orch: FakeOrchestrator
) -> None:
    store.save(make_job("cap", state=JobState.STOPPED_AT_CAP, cost_so_far=2.0, cost_cap=2.0))

    def refuse(job_id: str, new_cap: float) -> Job:
        raise ValueError("The new cap must be more than the $2.00 already spent.")

    orch.raise_cap_and_continue = refuse  # type: ignore[method-assign]
    window = make_window(ask_new_cap=lambda job: 2.5)
    window.select_job("cap")
    window.job_detail.raise_cap_button.click()
    assert "already spent" in window.job_detail.notice_label.text()
    assert orch.run_ids == []


def test_cancel_button_sets_the_cancel_event(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    orch.block = True
    window = make_window()
    window.new_job_panel.source_edit.setText(YOUTUBE)
    window.new_job_panel.play_button.click()
    assert orch.started.wait(5)
    qtbot.waitUntil(lambda: window.job_detail.cancel_button.isVisible(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    window.job_detail.cancel_button.click()
    qtbot.waitUntil(lambda: not window.has_running_jobs(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert window.job_detail.status_label.text() == "Cancelled"


def test_close_with_running_job_asks_and_can_refuse(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    orch.block = True
    answers = [False, True]
    asked: list[bool] = []

    def confirm() -> bool:
        answer = answers.pop(0)
        asked.append(answer)
        return answer

    window = make_window(confirm_close=confirm)
    window.new_job_panel.source_edit.setText(YOUTUBE)
    window.new_job_panel.play_button.click()
    assert orch.started.wait(5)

    window.close()
    assert window.isVisible()
    window.close()
    assert not window.isVisible()
    assert asked == [False, True]
    orch.release.set()
    qtbot.waitUntil(lambda: not window.has_running_jobs(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_close_without_running_job_does_not_ask(make_window: Callable[..., MainWindow]) -> None:
    asked: list[bool] = []
    window = make_window(confirm_close=lambda: asked.append(True) or True)
    window.close()
    assert asked == []
    assert not window.isVisible()


def test_alarm_shown_for_final_job_with_machine(
    make_window: Callable[..., MainWindow], store: JobStore
) -> None:
    store.save(make_job("alarm", state=JobState.FAILED, instance_id=4242, error="It failed."))
    window = make_window()
    item = window.job_item("alarm")
    assert item.alarm_label.isVisible()
    window.select_job("alarm")
    alarm = window.job_detail.alarm_label
    assert alarm.isVisible()
    assert "4242" in alarm.text()
    assert "cloud.vast.ai" in alarm.text()


HTML_NAME = '<b>x</b><a href="https://evil">click</a>'


def test_job_text_is_shown_as_plain_text_not_html(
    make_window: Callable[..., MainWindow], store: JobStore
) -> None:
    store.save(
        make_job(
            "html", name=HTML_NAME, state=JobState.FAILED, source="<i>src</i>",
            message="<img src=x>", error="<h1>boom</h1>",
        )
    )  # fmt: skip
    window = make_window()
    item = window.job_item("html")
    assert item.name_label.textFormat() == Qt.TextFormat.PlainText
    assert item.name_label.text() == HTML_NAME
    assert item.summary_label.textFormat() == Qt.TextFormat.PlainText
    assert "<b>" not in item.name_label.toolTip()  # escaped, so shown literally
    window.select_job("html")
    detail = window.job_detail
    for label, text in (
        (detail.title_label, HTML_NAME),
        (detail.source_label, "<i>src</i>"),
        (detail.message_label, "<h1>boom</h1>"),
    ):
        assert label.textFormat() == Qt.TextFormat.PlainText
        assert label.text() == text
    for label in (detail.status_label, detail.alarm_label, detail.notice_label, detail.cost_label):
        assert label.textFormat() == Qt.TextFormat.PlainText
    assert all(text.textFormat() == Qt.TextFormat.PlainText for text in detail.stage_text_labels())


class FakeDialog:
    def __init__(self, opened: list[bool]) -> None:
        self._opened = opened

    def exec(self) -> int:
        self._opened.append(True)
        return 0


def test_keys_button_is_disabled_while_a_job_runs(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    orch.block = True
    opened: list[bool] = []
    window = make_window(setup_dialog_factory=lambda parent: FakeDialog(opened))
    assert window.keys_button.isEnabled()
    window.new_job_panel.source_edit.setText(YOUTUBE)
    window.new_job_panel.play_button.click()
    assert orch.started.wait(5)
    assert not window.keys_button.isEnabled()
    assert window.keys_button.toolTip() == "Finish or cancel running jobs before changing keys"
    window.open_keys_dialog()  # the open path is guarded too, not only the button
    assert opened == []
    orch.release.set()
    qtbot.waitUntil(lambda: not window.has_running_jobs(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert window.keys_button.isEnabled()
    window.open_keys_dialog()
    assert opened == [True]


def test_keys_button_is_disabled_during_the_startup_check(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    orch.hold_startup = True
    opened: list[bool] = []
    window = make_window(run_startup=True, setup_dialog_factory=lambda parent: FakeDialog(opened))
    assert not window.keys_button.isEnabled()
    window.open_keys_dialog()
    assert opened == []
    orch.startup_gate.set()
    qtbot.waitUntil(lambda: not window.has_background_work(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert window.keys_button.isEnabled()


def test_cancelling_state_survives_job_updates(qtbot: object) -> None:
    detail = JobDetail(opener=lambda path: None)
    qtbot.addWidget(detail)  # type: ignore[attr-defined]
    detail.show()
    running = make_job("c", state=JobState.LABELING, chunk=1, chunks=4, percent=10)
    detail.show_job(running, running=True)
    with qtbot.waitSignal(detail.cancel_requested, timeout=WAIT_MS):  # type: ignore[attr-defined]
        detail.cancel_button.click()
    detail.show_job(running.with_(percent=20, chunk=2), running=True)
    assert not detail.cancel_button.isEnabled()
    assert detail.status_label.text().startswith("Cancelling")
    detail.show_job(running.with_(state=JobState.CANCELLED), running=False)
    assert detail.status_label.text() == "Cancelled"
    detail.show_job(make_job("c", state=JobState.LABELING), running=True)  # e.g. a later rerun
    assert detail.cancel_button.isEnabled()


def test_no_alarm_for_ordinary_failed_job(
    make_window: Callable[..., MainWindow], store: JobStore
) -> None:
    store.save(make_job("f", state=JobState.FAILED, error="No machine."))
    window = make_window()
    window.select_job("f")
    assert not window.job_detail.alarm_label.isVisible()
    assert not window.job_item("f").alarm_label.isVisible()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("boom"), "something went wrong"),
        (JobAlreadyRunningError("This job is already running."), "already running"),
    ],
)
def test_worker_error_is_shown_in_plain_words(
    qtbot: object,
    make_window: Callable[..., MainWindow],
    orch: FakeOrchestrator,
    error: BaseException,
    expected: str,
) -> None:
    orch.run_error = error
    window = make_window()
    window.new_job_panel.source_edit.setText(YOUTUBE)
    window.new_job_panel.play_button.click()
    qtbot.waitUntil(lambda: not window.has_running_jobs(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: expected in window.job_detail.notice_label.text().lower(), timeout=WAIT_MS
    )


def test_startup_tasks_run_off_the_ui_thread_in_order(
    qtbot: object, make_window: Callable[..., MainWindow], store: JobStore, orch: FakeOrchestrator
) -> None:
    store.save(make_job("resume", state=JobState.LABELING, instance_id=9))
    orch.resume_ids = ["resume"]
    window = make_window(run_startup=True)
    qtbot.waitUntil(lambda: orch.run_ids == ["resume"], timeout=WAIT_MS)  # type: ignore[attr-defined]
    main = threading.get_ident()
    assert orch.calls[:2] == ["resume_pending", "reap_orphans"]
    assert orch.threads["resume_pending"] != main
    assert orch.threads["reap_orphans"] != main
    qtbot.waitUntil(lambda: not window.has_background_work(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_play_during_startup_check_waits_until_it_finishes(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    orch.hold_startup = True
    window = make_window(run_startup=True)
    qtbot.waitUntil(lambda: "resume_pending" in orch.calls, timeout=WAIT_MS)  # type: ignore[attr-defined]
    window.new_job_panel.source_edit.setText(YOUTUBE)
    window.new_job_panel.play_button.click()
    assert "submit" not in orch.calls  # resume_pending must not see a half-started job
    assert "startup" in window.statusBar().currentMessage().lower()
    orch.startup_gate.set()
    qtbot.waitUntil(lambda: "run" in orch.calls, timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert orch.calls.index("submit") > orch.calls.index("reap_orphans")
    qtbot.waitUntil(lambda: not window.has_background_work(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def test_startup_failure_is_reported_not_raised(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    def broken() -> list[str]:
        raise RuntimeError("Vast.ai is down")

    orch.resume_pending = broken  # type: ignore[method-assign]
    window = make_window(run_startup=True)
    qtbot.waitUntil(lambda: "reap_orphans" in orch.calls, timeout=WAIT_MS)  # type: ignore[attr-defined]
    qtbot.waitUntil(lambda: window.startup_warning.isVisible(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert "could not" in window.startup_warning.text().lower()
    assert "unfinished jobs" in window.startup_warning.toolTip().lower()  # the detail
    window.stop_background_timers()  # the retry would run the broken check again


def test_retry_destroys_runs_periodically_off_the_ui_thread(
    qtbot: object, make_window: Callable[..., MainWindow], orch: FakeOrchestrator
) -> None:
    window = make_window(retry_interval_ms=20)
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: orch.calls.count("retry_unconfirmed_destroys") >= 2, timeout=WAIT_MS
    )
    assert orch.threads["retry_unconfirmed_destroys"] != threading.get_ident()
    window.stop_background_timers()


# --- setup dialog (more cases in test_gui_setup.py) ---------------------------------------------


def test_setup_dialog_rejects_invalid_token_without_saving(qtbot: object) -> None:
    backend = FakeBackend()
    creds = CredentialStore(backend=backend)
    dialog = SetupDialog(
        creds,
        check_vast=lambda key: 12.5,
        check_hf=lambda token: HfAccess(
            valid=False, user=None, models={DIARIZATION_MODEL: False, SEGMENTATION_MODEL: False}
        ),
    )
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.vast_edit.setText("vast-key-1234567890")
    dialog.hf_edit.setText("hf_token_abcdefghijkl")
    dialog.save_button.click()
    qtbot.waitUntil(lambda: not dialog.is_checking(), timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert backend.data == {}
    assert "did not accept" in dialog.hf_status.text()
    assert dialog.result() != SetupDialog.DialogCode.Accepted
