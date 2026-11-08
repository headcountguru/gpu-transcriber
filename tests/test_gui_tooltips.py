"""Tooltips of job text: shown exactly as written, never as HTML and never as HTML codes."""

from pathlib import Path

import pytest
from PySide6.QtGui import Qt, QTextDocument  # QtGui adds mightBeRichText to Qt

from app.credentials import CredentialStore
from app.gui.main_window import MainWindow, plain_tooltip
from app.jobs import JobState, JobStore
from tests.gui_fakes import FakeBackend, FakeOrchestrator, make_job

NAMES = ["Tom's & Co", "<b>bold</b> & 'quoted' \"text\"", "a < b > c", "plain"]


def rendered(tooltip: str) -> str:
    """What Qt shows for a tooltip: rich text is rendered, other text is shown as is."""
    if not Qt.mightBeRichText(tooltip):
        return tooltip
    document = QTextDocument()
    document.setHtml(tooltip)
    return document.toPlainText()


@pytest.mark.parametrize("text", NAMES)
def test_plain_tooltip_renders_the_original_text(qtbot: object, text: str) -> None:
    tooltip = plain_tooltip(text)

    assert Qt.mightBeRichText(tooltip)  # always rich, so the escapes are decoded by Qt
    assert rendered(tooltip) == text


def test_plain_tooltip_of_nothing_is_empty() -> None:
    assert plain_tooltip("") == ""


def test_job_row_tooltip_shows_apostrophe_and_ampersand(qtbot: object, tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    store.save(make_job("amp", name="Tom's & Co", state=JobState.FAILED, error="e"))
    orch = FakeOrchestrator(store)
    window = MainWindow(
        orch,
        store,
        CredentialStore(backend=FakeBackend()),
        run_startup=False,
        confirm_close=lambda: True,
        retry_interval_ms=0,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    try:
        assert rendered(window.job_item("amp").name_label.toolTip()) == "Tom's & Co"
    finally:
        orch.release.set()
        orch.startup_gate.set()
