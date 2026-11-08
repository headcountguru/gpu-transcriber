"""The "new job" panel: a link-or-file input, the time and cost estimate, and Play."""

import logging
import os
from collections.abc import Callable

from PySide6.QtCore import QMimeData, Qt, QTimer, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.estimate import estimate_job, format_estimate
from app.gui import theme
from app.gui.worker import TaskWorker
from app.presenter import SourceCheck, check_source

logger = logging.getLogger(__name__)

# A typical Vast.ai rate for a suitable GPU, used only for the estimate shown before Play. The
# real price is chosen when the machine is rented (the first live test paid $0.23/h for an
# RTX A5000), and the spending cap is set from that real price.
ESTIMATE_PRICE_PER_HOUR = 0.35

DEBOUNCE_MS = 600
PLAY_TOOLTIP = "Paste a YouTube link or choose a file"
EMPTY_STATE_TEXT = "Paste a YouTube link or choose an audio file, then press Play."
ESTIMATE_NOTE = (
    "Estimate only. The real time and price depend on the machine rented and are not yet "
    "confirmed for long recordings. A spending cap is set automatically."
)
CHECKING_TEXT = "Checking the length…"
FILE_FILTER = (
    "Audio or video files (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma *.mp4 *.mkv "
    "*.mov *.avi *.webm);;All files (*)"
)

Probe = Callable[[str, bool], float]
ChooseFile = Callable[[QWidget], str | None]


def _choose_file_dialog(parent: QWidget) -> str | None:
    path, _ = QFileDialog.getOpenFileName(parent, "Choose an audio or video file", "", FILE_FILTER)
    return path or None


def local_file_from(mime: QMimeData) -> str | None:
    """The first dropped item that is an existing local file, else ``None``."""
    if not mime.hasUrls():
        return None
    for url in mime.urls():
        if url.isLocalFile():
            path = os.path.normpath(url.toLocalFile())
            if os.path.isfile(path):
                return path
    return None


class NewJobPanel(QWidget):
    """Emits ``play_requested(source, is_youtube)`` when Play is pressed on a valid input."""

    play_requested = Signal(str, bool)

    def __init__(
        self,
        probe: Probe | None,
        parent: QWidget | None = None,
        *,
        debounce_ms: int = DEBOUNCE_MS,
        choose_file: ChooseFile = _choose_file_dialog,
    ) -> None:
        super().__init__(parent)
        self._probe = probe
        self._choose_file = choose_file
        self._check = check_source("")
        self._worker: TaskWorker | None = None
        self._probing_key: tuple[str, bool] | None = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._start_probe)
        self.setAcceptDrops(True)
        self._build()
        self._refresh()

    # --- layout ---------------------------------------------------------------------------------

    def _build(self) -> None:
        title = QLabel("New transcription")
        title.setObjectName("pageTitle")
        hint = QLabel(EMPTY_STATE_TEXT)
        hint.setObjectName("emptyState")
        hint.setWordWrap(True)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("YouTube link, or the path to an audio or video file")
        self.source_edit.setAcceptDrops(False)  # drops land on the panel, which takes the path
        self.source_edit.setClearButtonEnabled(True)
        self.source_edit.setAccessibleName("YouTube link or audio file")
        self.source_edit.textChanged.connect(self._on_text_changed)
        self.source_edit.returnPressed.connect(self._on_play)
        self.browse_button = QPushButton("Choose file…")
        self.browse_button.clicked.connect(self._on_browse)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.source_edit, 1)
        row.addWidget(self.browse_button)

        drop_hint = QLabel("You can also drag an audio or video file onto this panel.")
        drop_hint.setObjectName("hint")
        self.error_label = QLabel()
        self.error_label.setObjectName("errorText")
        self.error_label.setWordWrap(True)
        self.estimate_label = QLabel()
        self.estimate_label.setObjectName("estimate")
        self.estimate_note = QLabel(ESTIMATE_NOTE)
        self.estimate_note.setObjectName("estimateNote")
        self.estimate_note.setWordWrap(True)

        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("primary")
        self.play_button.setDefault(True)
        self.play_button.clicked.connect(self._on_play)
        play_row = QHBoxLayout()
        play_row.addWidget(self.play_button)
        play_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 32, 36, 32)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addSpacing(14)
        layout.addLayout(row)
        layout.addWidget(drop_hint)
        layout.addWidget(self.error_label)
        layout.addSpacing(12)
        layout.addWidget(self.estimate_label)
        layout.addWidget(self.estimate_note)
        layout.addSpacing(16)
        layout.addLayout(play_row)
        layout.addStretch(1)
        self.setTabOrder(self.source_edit, self.browse_button)
        self.setTabOrder(self.browse_button, self.play_button)

    # --- state ----------------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the input after a job was started."""
        self.source_edit.clear()

    def is_probing(self) -> bool:
        return self._worker is not None

    def focus_input(self) -> None:
        self.source_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def _refresh(self) -> None:
        check = self._check
        self.play_button.setEnabled(check.ok)
        self.play_button.setToolTip("" if check.ok else PLAY_TOOLTIP)
        self._show_error(check.error)
        if not check.ok:
            self.estimate_label.hide()
            self.estimate_note.hide()

    def _show_error(self, message: str | None) -> None:
        self.error_label.setText(message or "")
        self.error_label.setVisible(bool(message))

    def _on_text_changed(self, text: str) -> None:
        self._check = check_source(text)
        self._refresh()
        if self._check.ok and self._probe is not None:
            self.estimate_label.setText(CHECKING_TEXT)
            theme.set_kind(self.estimate_label, theme.KIND_MUTED)
            self.estimate_label.show()
            self.estimate_note.hide()
            self._debounce.start()
        else:
            self._debounce.stop()

    def _on_browse(self) -> None:
        path = self._choose_file(self)
        if path:
            self.source_edit.setText(os.path.normpath(path))

    def _on_play(self) -> None:
        if not self._check.ok:
            return
        self.play_requested.emit(self._check.source, self._check.is_youtube)

    # --- estimate -------------------------------------------------------------------------------

    def _key(self, check: SourceCheck) -> tuple[str, bool]:
        return check.source, check.is_youtube

    def _start_probe(self) -> None:
        probe = self._probe
        if probe is None or not self._check.ok or self._worker is not None:
            return  # a running probe re-checks the latest input when it finishes
        source, is_youtube = self._probing_key = self._key(self._check)
        worker = TaskWorker(lambda: probe(source, is_youtube), "Length check", self)
        worker.succeeded.connect(self._on_probe_result)
        worker.failed.connect(self._on_probe_failed)
        worker.finished.connect(self._on_probe_finished)
        self._worker = worker
        worker.start()

    def _is_current(self) -> bool:
        return self._check.ok and self._probing_key == self._key(self._check)

    def _on_probe_result(self, duration_s: object) -> None:
        if not self._is_current() or not isinstance(duration_s, int | float):
            return
        text = format_estimate(estimate_job(float(duration_s), ESTIMATE_PRICE_PER_HOUR))
        self.estimate_label.setText(text)
        theme.set_kind(self.estimate_label, "")
        self.estimate_label.show()
        self.estimate_note.show()

    def _on_probe_failed(self, message: str) -> None:
        if not self._is_current():
            return
        self.estimate_label.hide()
        self.estimate_note.hide()
        self._show_error(message)

    def _on_probe_finished(self) -> None:
        worker = self._worker
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self._worker = None
        if self._check.ok and not self._is_current():
            self._start_probe()

    # --- drag and drop --------------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if local_file_from(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        path = local_file_from(event.mimeData())
        if path is None:
            event.ignore()
            return
        self.source_edit.setText(path)
        event.acceptProposedAction()
