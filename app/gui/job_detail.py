"""The selected job: name, status, stages, progress, cost, and what can be done next."""

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from app.gui import theme
from app.gui.new_job_panel import EMPTY_STATE_TEXT
from app.jobs import TERMINAL_STATES, Job, JobState, output_file_name
from app.presenter import (
    alarm_text,
    cost_text,
    stage_rows,
    status_kind,
    suggested_cap,
    summary_line,
)

logger = logging.getLogger(__name__)

TRANSCRIPT_FILE = "transcript.txt"
TIMESTAMPED_FILE = "transcript_timestamped.txt"
SUBTITLES_FILE = "transcript.srt"
STAGE_COUNT = 6
CANCELLING_TEXT = "Cancelling — shutting the machine down…"
MAX_CAP_DOLLARS = 1000.0

_STAGE_KIND = {
    "done": theme.KIND_DONE,
    "active": theme.KIND_RUNNING,
    "pending": theme.KIND_MUTED,
    "failed": theme.KIND_FAILED,
}

Opener = Callable[[str], None]


def _label(object_name: str = "", text: str = "", wrap: bool = True) -> QLabel:
    label = QLabel(text)
    # Job names, sources and messages come from YouTube titles, files and the remote machine:
    # never let Qt guess they are rich text (a title could otherwise render links or markup).
    label.setTextFormat(Qt.TextFormat.PlainText)
    if object_name:
        label.setObjectName(object_name)
    label.setWordWrap(wrap)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


class JobDetail(QWidget):
    """Right-hand side of the window. ``show_job(None)`` shows the empty state."""

    cancel_requested = Signal(str)
    raise_cap_requested = Signal(str)

    def __init__(self, opener: Opener, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._opener = opener
        self._job: Job | None = None
        self._stage_status: dict[str, str] = {}
        self._cancelling: set[str] = set()
        self._stack = QStackedLayout(self)
        self._stack.addWidget(self._build_empty())
        self._stack.addWidget(self._build_content())
        self.show_job(None)

    # --- layout ---------------------------------------------------------------------------------

    def _build_empty(self) -> QWidget:
        page = QWidget()
        self.empty_label = _label("emptyState", EMPTY_STATE_TEXT)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(page)
        layout.addWidget(self.empty_label)
        return page

    def _build_content(self) -> QWidget:
        body = QWidget()
        body.setObjectName("detailBody")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(8)
        self.title_label = _label("pageTitle")
        self.status_label = _label("statusLine")
        self.source_label = _label("source")
        self.alarm_label = _label("alarm")
        self.message_label = _label()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        for widget in (self.title_label, self.status_label, self.source_label):
            layout.addWidget(widget)
        layout.addSpacing(8)
        layout.addWidget(self.alarm_label)
        layout.addWidget(self.message_label)
        layout.addSpacing(6)
        layout.addWidget(self.progress_bar)
        layout.addSpacing(14)
        layout.addWidget(_label("sectionTitle", "Stages"))
        layout.addLayout(self._build_stages())
        layout.addSpacing(10)
        self.cost_label = _label("costLabel")
        layout.addWidget(self.cost_label)
        layout.addSpacing(12)
        layout.addLayout(self._build_actions())
        self.outputs_title = _label("sectionTitle", "Transcript files")
        layout.addWidget(self.outputs_title)
        layout.addLayout(self._build_outputs())
        layout.addSpacing(8)
        self.notice_label = _label("notice")
        layout.addWidget(self.notice_label)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("detailScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(body)
        return scroll

    def _build_stages(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        self._stage_markers: list[QLabel] = []
        self._stage_labels: list[QLabel] = []
        for row in range(STAGE_COUNT):
            marker = _label("stageMarker", wrap=False)
            marker.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text = _label()
            grid.addWidget(marker, row, 0)
            grid.addWidget(text, row, 1)
            self._stage_markers.append(marker)
            self._stage_labels.append(text)
        return grid

    def _build_actions(self) -> QHBoxLayout:
        self.cancel_button = QPushButton("Cancel job")
        self.cancel_button.setToolTip("Stop the job and shut the rented machine down")
        self.cancel_button.clicked.connect(self._on_cancel)
        self.raise_cap_button = QPushButton("Raise cap and continue")
        self.raise_cap_button.setObjectName("warning")
        self.raise_cap_button.setToolTip("Choose a higher spending cap and continue")
        self.raise_cap_button.clicked.connect(self._on_raise_cap)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.cancel_button)
        row.addWidget(self.raise_cap_button)
        row.addStretch(1)
        return row

    def _build_outputs(self) -> QHBoxLayout:
        self.open_transcript_button = QPushButton("Open transcript")
        self.open_timestamped_button = QPushButton("Open timestamped")
        self.open_subtitles_button = QPushButton("Open subtitles")
        self.show_folder_button = QPushButton("Show in folder")
        self.open_transcript_button.clicked.connect(lambda: self._open_output(TRANSCRIPT_FILE))
        self.open_timestamped_button.clicked.connect(lambda: self._open_output(TIMESTAMPED_FILE))
        self.open_subtitles_button.clicked.connect(lambda: self._open_output(SUBTITLES_FILE))
        self.show_folder_button.clicked.connect(self._show_folder)
        row = QHBoxLayout()
        row.setSpacing(8)
        for button in self._output_buttons():
            row.addWidget(button)
        row.addStretch(1)
        return row

    def _output_buttons(self) -> tuple[QPushButton, ...]:
        return (
            self.open_transcript_button,
            self.open_timestamped_button,
            self.open_subtitles_button,
            self.show_folder_button,
        )

    # --- showing a job --------------------------------------------------------------------------

    @property
    def job(self) -> Job | None:
        return self._job

    def show_job(self, job: Job | None, running: bool = False, notice: str = "") -> None:
        """Show ``job`` (or the empty state). ``running`` means a worker is driving it now."""
        self._job = job
        if job is None:
            self._stack.setCurrentIndex(0)
            return
        self._stack.setCurrentIndex(1)
        if job.state in TERMINAL_STATES:
            self._cancelling.discard(job.id)
        self._show_header(job)
        self._show_progress(job)
        self._show_stages(job)
        self.cost_label.setText(f"Cost: {cost_text(job)}")
        self._show_buttons(job, running)
        self.show_notice(notice)

    def show_notice(self, text: str) -> None:
        self.notice_label.setText(text)
        self.notice_label.setVisible(bool(text))

    def _show_header(self, job: Job) -> None:
        self.title_label.setText(job.name)
        cancelling = job.id in self._cancelling
        self.status_label.setText(CANCELLING_TEXT if cancelling else summary_line(job))
        theme.set_kind(self.status_label, status_kind(job))
        self.source_label.setText(job.source)
        alarm = alarm_text(job)
        self.alarm_label.setText(alarm or "")
        self.alarm_label.setVisible(alarm is not None)
        failed = job.state is JobState.FAILED
        message = (job.error or job.message) if failed else job.message
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))
        theme.set_kind(self.message_label, theme.KIND_FAILED if failed else theme.KIND_MUTED)

    def _show_progress(self, job: Job) -> None:
        percent = 100 if job.state is JobState.DONE else max(0, min(100, job.percent))
        self.progress_bar.setValue(percent)
        theme.set_kind(self.progress_bar, status_kind(job))

    def _show_stages(self, job: Job) -> None:
        self._stage_status = {}
        for row, marker, text in zip(
            stage_rows(job), self._stage_markers, self._stage_labels, strict=True
        ):
            kind = _STAGE_KIND[row.status]
            marker.setText(theme.STAGE_MARKERS[row.status])
            marker.setToolTip(row.status.capitalize())
            text.setText(row.label)
            theme.set_kind(marker, kind)
            theme.set_kind(text, theme.KIND_MUTED if row.status == "pending" else "")
            font = text.font()
            font.setBold(row.status == "active")
            text.setFont(font)
            self._stage_status[row.label] = row.status

    def _show_buttons(self, job: Job, running: bool) -> None:
        self.cancel_button.setVisible(running and job.state not in TERMINAL_STATES)
        self.cancel_button.setEnabled(job.id not in self._cancelling)
        self.raise_cap_button.setVisible(job.state is JobState.STOPPED_AT_CAP and not running)
        has_outputs = job.state is JobState.DONE and bool(job.output_dir)
        self.outputs_title.setVisible(has_outputs)
        for button in self._output_buttons():
            button.setVisible(has_outputs)

    def stage_text_labels(self) -> list[QLabel]:
        return list(self._stage_labels)

    def stage_texts(self) -> list[str]:
        return [label.text() for label in self._stage_labels]

    def stage_status(self, label: str) -> str | None:
        return self._stage_status.get(label)

    # --- actions --------------------------------------------------------------------------------

    def _on_cancel(self) -> None:
        if self._job is None:
            return
        self._cancelling.add(self._job.id)  # kept until the job reaches a final state
        self.cancel_button.setEnabled(False)
        self.status_label.setText(CANCELLING_TEXT)
        self.cancel_requested.emit(self._job.id)

    def _on_raise_cap(self) -> None:
        if self._job is not None:
            self.raise_cap_requested.emit(self._job.id)

    def _output_dir(self) -> Path | None:
        if self._job is None or not self._job.output_dir:
            return None
        return Path(self._job.output_dir)

    def _open(self, path: Path, what: str) -> None:
        try:
            self._opener(str(path))
        except OSError as exc:
            logger.warning("Could not open %s: %s", path, exc)
            self.show_notice(f"Windows could not open the {what}. Try opening it from the folder.")
            return
        self.show_notice("")

    def _output_path(self, remote_name: str) -> Path | None:
        """The finished file: ``<title>-<remote_name>``, or the plain name for older jobs."""
        folder = self._output_dir()
        if folder is None or self._job is None:
            return None
        named = folder / output_file_name(self._job.name, remote_name)
        if named.is_file():
            return named
        legacy = folder / remote_name
        return legacy if legacy.is_file() else named

    def _open_output(self, name: str) -> None:
        path = self._output_path(name)
        if path is None or not path.is_file():
            shown = output_file_name(self._job.name, name) if self._job else name
            self.show_notice(f"Can't find {shown}. It may have been moved or deleted.")
            return
        self._open(path, path.name)

    def _show_folder(self) -> None:
        folder = self._output_dir()
        if folder is None or not folder.is_dir():
            self.show_notice("Can't find the job's folder. It may have been moved or deleted.")
            return
        self._open(folder, "folder")


def ask_new_cap_dialog(parent: QWidget, job: Job) -> float | None:
    """Ask for a new spending cap in dollars; ``None`` if the user cancels."""
    minimum, default = suggested_cap(job)
    dialog = QDialog(parent)
    dialog.setWindowTitle("Raise the spending cap")
    text = QLabel(
        f"This job stopped at its spending cap of ${job.cost_cap:.2f} after spending "
        f"${max(job.cost_so_far, job.banked_cost):.2f}. The finished parts are saved.\n\n"
        "Choose a new cap. The job continues from the saved parts and stops again if the new "
        "cap is reached."
    )
    text.setWordWrap(True)
    spin = QDoubleSpinBox()
    spin.setPrefix("$")
    spin.setDecimals(2)
    spin.setSingleStep(0.5)
    spin.setRange(minimum, max(MAX_CAP_DOLLARS, default))
    spin.setValue(default)
    spin.setAccessibleName("New spending cap in dollars")
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Raise cap and continue")
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(12)
    layout.addWidget(text)
    layout.addWidget(spin)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(420)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return float(spin.value())
