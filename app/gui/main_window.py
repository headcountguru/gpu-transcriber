"""Main window, layout B: the job list on the left, the selected job (or a new one) on the right.

Every slow call runs on a worker thread: job runs (``JobWorker``), the start-up check
(``resume_pending`` then ``reap_orphans``) and the periodic ``retry_unconfirmed_destroys``.
Jobs started while the start-up check runs wait for it, so ``resume_pending`` never sees a job
this window is about to run.
"""

import html
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.credentials import CredentialStore
from app.gui import theme
from app.gui.job_detail import JobDetail, Opener, ask_new_cap_dialog
from app.gui.new_job_panel import NewJobPanel, Probe
from app.gui.worker import JobWorker, TaskWorker, plain_error
from app.jobs import Job, JobStore
from app.presenter import alarm_text, status_kind, summary_line
from common.protocol import GRACE_AFTER_FINISH_S

logger = logging.getLogger(__name__)

RETRY_INTERVAL_MS = 5 * 60 * 1000
SIDEBAR_WIDTH = 300
ROW_EXTRA_PX = 4  # the row's bottom border would otherwise clip descenders (g, p, y)
MIN_WIDTH = 900
MIN_HEIGHT = 600
STATUS_TIMEOUT_MS = 0  # keep messages until replaced
CLOSE_TITLE = "A job is still running"
_GRACE_HOURS = f"{GRACE_AFTER_FINISH_S / 3600:.0f} hours"
CLOSE_TEXT = (
    "A job is still running. If you close the app now, the rented machine keeps running and "
    "charging (usually about $0.20-$0.60 an hour) until the job finishes. After that it waits "
    f"up to {_GRACE_HOURS} for the app to come back and download the results, and keeps "
    "charging while it waits. Its safety timer shuts it down after its time limit (at least "
    f"{_GRACE_HOURS}). The spending cap is enforced only while the app is open. "
    "Reopen the app to download your results and shut the machine down. Close anyway?"
)
MSG_STARTUP_CHECK = "Checking for unfinished jobs and leftover machines…"
MSG_WAITING_FOR_STARTUP = "The job will start as soon as the startup check finishes."
KEYS_TOOLTIP = "Change the Vast.ai API key or the Hugging Face token"
KEYS_LOCKED_TOOLTIP = "Finish or cancel running jobs before changing keys"
MSG_JOB_FILE_MISSING = "This job's saved file is missing or damaged."


class Orchestrating(Protocol):
    """The part of ``Orchestrator`` the window uses."""

    def submit(self, source: str, is_youtube: bool, name: str | None = None) -> Job: ...

    def run(
        self, job_id: str, on_update: Callable[[Job], None], cancel: threading.Event | None = None
    ) -> Job: ...

    def resume_pending(self) -> list[str]: ...

    def reap_orphans(self) -> list[int]: ...

    def retry_unconfirmed_destroys(self) -> list[Job]: ...

    def raise_cap_and_continue(self, job_id: str, new_cap: float) -> Job: ...


@dataclass(frozen=True)
class StartupResult:
    resumed: tuple[str, ...]
    problems: tuple[str, ...]


def run_startup_tasks(orchestrator: Orchestrating) -> StartupResult:
    """Re-attach unfinished jobs, then remove orphaned machines. Blocking: worker thread only."""
    resumed: list[str] = []
    problems: list[str] = []
    try:
        resumed = orchestrator.resume_pending()
    except Exception as exc:
        logger.warning("Resuming jobs failed (%s)", type(exc).__name__)
        problems.append(f"Could not check for unfinished jobs: {plain_error(exc)}")
    try:
        orchestrator.reap_orphans()
    except Exception as exc:
        logger.warning("Removing leftover machines failed (%s)", type(exc).__name__)
        problems.append(f"Could not check for leftover machines: {plain_error(exc)}")
    return StartupResult(tuple(resumed), tuple(problems))


def open_with_windows(path: str) -> None:
    """Open a file or folder with its default Windows program (Explorer for a folder)."""
    os.startfile(path)  # type: ignore[attr-defined]  # Windows only


def confirm_close_dialog(parent: QWidget) -> bool:
    answer = QMessageBox.question(
        parent,
        CLOSE_TITLE,
        CLOSE_TEXT,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


class ElidedLabel(QLabel):
    """A plain-text label that keeps its full text but paints it cut short with "…" if needed."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, rect.width())
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(
            rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text
        )


def plain_tooltip(text: str) -> str:
    """Job text as a tooltip that Qt shows exactly as written.

    Tooltips guess rich text like labels do, and only text that looks like HTML is rendered.
    Wrapping the escaped text in a paragraph makes Qt always render it, so ``&amp;`` and
    ``&#x27;`` show as ``&`` and ``'`` and a name like ``<b>x</b>`` shows literally.
    """
    if not text:
        return ""
    return f'<p style="white-space:pre-wrap">{html.escape(text)}</p>'


class JobListItem(QWidget):
    """One row in the job list: name, short status, and the machine alarm when needed."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.name_label = ElidedLabel()
        self.name_label.setObjectName("jobName")
        self.summary_label = ElidedLabel()
        self.summary_label.setObjectName("jobSummary")
        self.alarm_label = QLabel("⚠ Machine may still be running")
        self.alarm_label.setTextFormat(Qt.TextFormat.PlainText)
        self.alarm_label.setObjectName("jobAlarm")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 12, 10)
        layout.setSpacing(2)
        layout.addWidget(self.name_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.alarm_label)

    def show_job(self, job: Job) -> None:
        self.name_label.setText(job.name)
        self.name_label.setToolTip(plain_tooltip(job.name))
        summary = summary_line(job)
        self.summary_label.setText(summary)
        self.summary_label.setToolTip(plain_tooltip(summary))
        theme.set_kind(self.summary_label, status_kind(job))
        alarm = alarm_text(job)
        self.alarm_label.setVisible(alarm is not None)
        self.alarm_label.setToolTip(plain_tooltip(alarm or ""))


class MainWindow(QMainWindow):
    """The app's window. Collaborators are injected so tests never launch or rent anything."""

    def __init__(
        self,
        orchestrator: Orchestrating,
        store: JobStore,
        creds: CredentialStore,
        *,
        probe: Probe | None = None,
        opener: Opener = open_with_windows,
        setup_dialog_factory: Callable[[QWidget], QDialog] | None = None,
        confirm_close: Callable[[], bool] | None = None,
        ask_new_cap: Callable[[Job], float | None] | None = None,
        run_startup: bool = True,
        retry_interval_ms: int = RETRY_INTERVAL_MS,
    ) -> None:
        super().__init__()
        self._orch = orchestrator
        self._store = store
        self._creds = creds
        self._setup_dialog_factory = setup_dialog_factory
        self._confirm_close = confirm_close or (lambda: confirm_close_dialog(self))
        self._ask_new_cap = ask_new_cap or (lambda job: ask_new_cap_dialog(self, job))
        self._jobs: dict[str, Job] = {}
        self._items: dict[str, QListWidgetItem] = {}
        self._item_widgets: dict[str, JobListItem] = {}
        self._workers: dict[str, JobWorker] = {}
        self._notices: dict[str, str] = {}
        self._startup_worker: TaskWorker | None = None
        self._retry_worker: TaskWorker | None = None
        self._deferred: list[tuple[str, bool]] = []
        self._pending_resumed: tuple[str, ...] = ()
        self.setWindowTitle("GPU Transcriber")
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)
        self.setStyleSheet(theme.STYLESHEET)
        self._build(probe, opener)
        for job in store.load_all():
            self._upsert(job)
        self._retry_timer = QTimer(self)
        self._retry_timer.setInterval(retry_interval_ms)
        self._retry_timer.timeout.connect(self._start_retry)
        if retry_interval_ms > 0:
            self._retry_timer.start()
        if run_startup:
            self._start_startup()

    # --- layout ---------------------------------------------------------------------------------

    def _build(self, probe: Probe | None, opener: Opener) -> None:
        self.new_job_panel = NewJobPanel(probe)
        self.new_job_panel.play_requested.connect(self._on_play)
        self.job_detail = JobDetail(opener)
        self.job_detail.cancel_requested.connect(self._on_cancel)
        self.job_detail.raise_cap_requested.connect(self._on_raise_cap)
        self._pages = QStackedWidget()
        self._pages.addWidget(self.new_job_panel)
        self._pages.addWidget(self.job_detail)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        layout.addWidget(self._pages, 1)
        self.setCentralWidget(central)
        self.setTabOrder(self.new_button, self.job_list)
        self.setTabOrder(self.job_list, self.keys_button)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        title = QLabel("Jobs")
        title.setObjectName("sidebarTitle")
        self.new_button = QPushButton("+ New")
        self.new_button.setObjectName("primary")
        self.new_button.setToolTip("Start a new transcription")
        self.new_button.clicked.connect(self.show_new_job)
        header = QHBoxLayout()
        header.setContentsMargins(16, 16, 16, 10)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.new_button)

        self.job_list = QListWidget()
        self.job_list.setAccessibleName("Jobs")
        self.job_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.job_list.currentItemChanged.connect(self._on_current_changed)
        self.keys_button = QPushButton("Accounts and keys…")
        self.keys_button.setToolTip(KEYS_TOOLTIP)
        self.keys_button.clicked.connect(self.open_keys_dialog)
        self.keys_button.setVisible(self._setup_dialog_factory is not None)
        footer = QHBoxLayout()
        footer.setContentsMargins(16, 10, 16, 14)
        footer.addWidget(self.keys_button)
        footer.addStretch(1)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 1, 0)  # leave the 1px right border visible
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self.job_list, 1)
        layout.addLayout(footer)
        return sidebar

    # --- the job list ---------------------------------------------------------------------------

    def job_item(self, job_id: str) -> JobListItem:
        return self._item_widgets[job_id]

    def _upsert(self, job: Job, at_top: bool = False) -> None:
        self._jobs[job.id] = job
        widget = self._item_widgets.get(job.id)
        if widget is None:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, job.id)
            widget = JobListItem()
            if at_top:
                self.job_list.insertItem(0, item)
            else:
                self.job_list.addItem(item)
            self.job_list.setItemWidget(item, widget)
            self._items[job.id] = item
            self._item_widgets[job.id] = widget
        for label in widget.findChildren(QLabel):
            label.ensurePolished()  # stylesheet fonts first, so eliding and row height fit
        widget.show_job(job)
        # Width 0: the row takes the list's width, so no horizontal scrolling.
        height = widget.sizeHint().height() + ROW_EXTRA_PX
        self._items[job.id].setSizeHint(QSize(0, height))
        if self._selected_id() == job.id:
            self._refresh_detail()

    def _selected_id(self) -> str | None:
        item = self.job_list.currentItem()
        if item is None or self._pages.currentWidget() is not self.job_detail:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, str) else None

    def select_job(self, job_id: str) -> None:
        item = self._items.get(job_id)
        if item is None:
            return
        self.job_list.setCurrentItem(item)
        self._pages.setCurrentWidget(self.job_detail)
        self._refresh_detail()

    def show_new_job(self) -> None:
        self.job_list.setCurrentItem(None)
        self._pages.setCurrentWidget(self.new_job_panel)
        self.new_job_panel.focus_input()

    def _on_current_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            return
        self._pages.setCurrentWidget(self.job_detail)
        self._refresh_detail()

    def _refresh_detail(self) -> None:
        job_id = self._selected_id()
        job = self._jobs.get(job_id) if job_id else None
        if job is None:
            self.job_detail.show_job(None)
            return
        notice = self._notices.get(job.id, "")
        self.job_detail.show_job(job, running=job.id in self._workers, notice=notice)

    def _reload_from_store(self) -> None:
        """Pick up changes made by background tasks; jobs being run are left to their worker."""
        for job in self._store.load_all():
            if job.id not in self._workers:
                self._upsert(job)

    # --- running jobs ---------------------------------------------------------------------------

    def has_running_jobs(self) -> bool:
        return bool(self._workers)

    def has_background_work(self) -> bool:
        return (
            bool(self._workers)
            or self._startup_worker is not None
            or self._retry_worker is not None
            or self.new_job_panel.is_probing()
        )

    def _on_play(self, source: str, is_youtube: bool) -> None:
        if self._startup_worker is not None:
            self._deferred.append((source, is_youtube))
            self.new_job_panel.reset()
            self.statusBar().showMessage(MSG_WAITING_FOR_STARTUP, STATUS_TIMEOUT_MS)
            return
        self._submit(source, is_youtube)

    def _submit(self, source: str, is_youtube: bool) -> None:
        try:
            job = self._orch.submit(source, is_youtube)
        except Exception as exc:
            logger.warning("Could not create the job (%s)", type(exc).__name__)
            self.statusBar().showMessage(f"Could not start the job: {plain_error(exc)}")
            return
        self.new_job_panel.reset()
        self._upsert(job, at_top=True)
        self.select_job(job.id)
        self._start_job(job.id)

    def _start_job(self, job_id: str) -> None:
        if job_id in self._workers:
            return
        self._notices.pop(job_id, None)
        worker = JobWorker(self._orch, job_id, self)
        worker.job_updated.connect(self._on_job_updated)
        worker.ended.connect(self._on_job_ended)
        self._workers[job_id] = worker
        worker.start()
        self._update_keys_button()
        self._refresh_detail()

    def _on_job_updated(self, job: object) -> None:
        if isinstance(job, Job):
            self._upsert(job)

    def _on_job_ended(self, job_id: str, message: str) -> None:
        worker = self._workers.pop(job_id, None)
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self._update_keys_button()
        if message:
            self._notices[job_id] = message
        try:
            self._upsert(self._store.get(job_id))
        except (FileNotFoundError, ValueError):
            logger.warning("Job %s could not be re-read after its run", job_id)
        self._refresh_detail()

    def _on_cancel(self, job_id: str) -> None:
        worker = self._workers.get(job_id)
        if worker is not None:
            worker.cancel()

    def _on_raise_cap(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None or job_id in self._workers:
            return
        new_cap = self._ask_new_cap(job)
        if new_cap is None:
            return
        try:
            updated = self._orch.raise_cap_and_continue(job_id, new_cap)
        except ValueError as exc:
            self._notices[job_id] = str(exc)
            self._refresh_detail()
            return
        except Exception as exc:
            logger.warning("Raising the cap failed (%s)", type(exc).__name__)
            self._notices[job_id] = (
                MSG_JOB_FILE_MISSING if isinstance(exc, FileNotFoundError) else plain_error(exc)
            )
            self._refresh_detail()
            return
        self._upsert(updated)
        self._start_job(job_id)

    # --- background checks ----------------------------------------------------------------------

    def _start_startup(self) -> None:
        orchestrator = self._orch
        worker = TaskWorker(lambda: run_startup_tasks(orchestrator), "Startup check", self)
        worker.succeeded.connect(self._on_startup_done)
        worker.failed.connect(self._on_background_failed)
        worker.finished.connect(self._on_startup_finished)
        self._startup_worker = worker
        self.statusBar().showMessage(MSG_STARTUP_CHECK, STATUS_TIMEOUT_MS)
        worker.start()
        self._update_keys_button()

    def _on_startup_done(self, result: object) -> None:
        if not isinstance(result, StartupResult):
            return
        self._reload_from_store()
        if result.problems:
            self.statusBar().showMessage(" ".join(result.problems), STATUS_TIMEOUT_MS)
        else:
            self.statusBar().clearMessage()
        self._pending_resumed = result.resumed

    def _on_startup_finished(self) -> None:
        worker = self._startup_worker
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self._startup_worker = None
        for job_id in self._pending_resumed:
            self._start_job(job_id)
        self._pending_resumed = ()
        deferred, self._deferred = self._deferred, []
        for source, is_youtube in deferred:
            self._submit(source, is_youtube)
        self._update_keys_button()

    def _start_retry(self) -> None:
        if self._retry_worker is not None or self._startup_worker is not None:
            return
        worker = TaskWorker(self._orch.retry_unconfirmed_destroys, "Machine re-check", self)
        worker.succeeded.connect(self._on_retry_done)
        worker.failed.connect(self._on_background_failed)
        worker.finished.connect(self._on_retry_finished)
        self._retry_worker = worker
        worker.start()
        self._update_keys_button()

    def _on_retry_done(self, _settled: object) -> None:
        self._reload_from_store()

    def _on_retry_finished(self) -> None:
        worker = self._retry_worker
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self._retry_worker = None
        self._update_keys_button()

    def _on_background_failed(self, message: str) -> None:
        self.statusBar().showMessage(message, STATUS_TIMEOUT_MS)

    def stop_background_timers(self) -> None:
        self._retry_timer.stop()

    # --- keys and closing -----------------------------------------------------------------------

    def _keys_locked(self) -> bool:
        """True while anything may be talking to Vast.ai with the current key.

        Switching to a key for another Vast.ai account mid-job would make the destroy of the
        running machine fail and leave it billing, so keys can only change when all is idle.
        """
        return (
            bool(self._workers)
            or self._startup_worker is not None
            or self._retry_worker is not None
        )

    def _update_keys_button(self) -> None:
        locked = self._keys_locked()
        self.keys_button.setEnabled(not locked)
        self.keys_button.setToolTip(KEYS_LOCKED_TOOLTIP if locked else KEYS_TOOLTIP)

    def open_keys_dialog(self) -> None:
        if self._setup_dialog_factory is None:
            return
        if self._keys_locked():  # guard the open path itself, not only the button
            self.statusBar().showMessage(KEYS_LOCKED_TOOLTIP, STATUS_TIMEOUT_MS)
            return
        self._retry_timer.stop()  # no background Vast.ai call while the key may change
        try:
            dialog = self._setup_dialog_factory(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.statusBar().showMessage("Keys checked and saved.", STATUS_TIMEOUT_MS)
        finally:
            if self._retry_timer.interval() > 0:
                self._retry_timer.start()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.has_running_jobs() and not self._confirm_close():
            event.ignore()
            return
        self.stop_background_timers()
        event.accept()
