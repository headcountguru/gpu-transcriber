"""Colors, fonts and the one stylesheet for the window.

Dark theme. Colors are semantic: red only for failures, amber for caps and alarms, green for
done, blue for the one primary action and for what is in progress. Widgets pick a color by
setting the ``kind`` property (see ``set_kind``) instead of inline styles.
"""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

BASE_POINT_SIZE = 10
FONT_FAMILY = "Segoe UI"

# Dark theme (GitHub-dark inspired). Text colors are contrast-checked in tests/test_theme.py.
INK = "#e6edf3"
MUTED = "#9da7b3"
FAINT = "#7d8590"
LINE = "#30363d"
SIDEBAR = "#161b22"
SURFACE = "#0d1117"
ACCENT = "#4493f8"
ACCENT_SOFT = "#12304f"
PRIMARY = "#1f6feb"
PRIMARY_HOVER = "#1a5fd0"
PRIMARY_DISABLED = "#1b2f4d"
PRIMARY_TEXT = "#ffffff"
HOVER = "#21262d"
PRESSED = "#2d333b"
TRACK = "#21262d"
GREEN = "#3fb950"
RED = "#ff7b72"
RED_SOFT = "#3d1418"
AMBER = "#e3b341"
AMBER_SOFT = "#3a2d0a"
AMBER_LINE = "#9e6a03"
AMBER_HOVER = "#4b3a0c"

# Kinds understood by the stylesheet's [kind="..."] selectors.
KIND_RUNNING = "running"
KIND_DONE = "done"
KIND_FAILED = "failed"
KIND_CAP = "cap"
KIND_MUTED = "muted"

STAGE_MARKERS = {"done": "✓", "active": "●", "pending": "○", "failed": "✕"}

STYLESHEET = f"""
QWidget {{ color: {INK}; font-family: "{FONT_FAMILY}"; font-size: {BASE_POINT_SIZE}pt; }}
QMainWindow, QDialog {{ background: {SURFACE}; }}
QScrollArea#detailScroll, QWidget#detailBody {{ background: {SURFACE}; }}
QWidget#sidebar {{ background: {SIDEBAR}; border-right: 1px solid {LINE}; }}
QLabel#sidebarTitle {{ font-size: 13pt; font-weight: 600; }}
QLabel#pageTitle {{ font-size: 18pt; font-weight: 600; }}
QLabel#sectionTitle {{ font-size: 11pt; font-weight: 600; color: {MUTED}; }}
QLabel#emptyState {{ font-size: 12pt; color: {MUTED}; }}
QLabel#hint, QLabel#source {{ color: {MUTED}; }}
QLabel#estimate {{ font-size: 12pt; font-weight: 600; }}
QLabel#estimateNote {{ color: {FAINT}; font-size: 9pt; }}
QLabel#costLabel {{ font-size: 11pt; }}
QLabel#jobName {{ font-weight: 600; }}
QLabel#jobSummary {{ color: {MUTED}; }}
QLabel#stageMarker {{ font-size: 12pt; min-width: 22px; }}

QLabel[kind="running"] {{ color: {ACCENT}; }}
QLabel[kind="done"] {{ color: {GREEN}; }}
QLabel[kind="failed"] {{ color: {RED}; }}
QLabel[kind="cap"] {{ color: {AMBER}; }}
QLabel[kind="muted"] {{ color: {FAINT}; }}

QLabel#alarm {{
    background: {AMBER_SOFT}; border: 1px solid {AMBER_LINE}; border-radius: 6px;
    color: {INK}; padding: 10px 12px;
}}
QLabel#jobAlarm {{ color: {AMBER}; font-size: 9pt; font-weight: 600; }}
QLabel#errorText {{ color: {RED}; }}
QLabel#notice {{
    background: {RED_SOFT}; border-radius: 6px; padding: 8px 12px; color: {RED};
}}

QLineEdit, QAbstractSpinBox {{
    border: 1px solid {LINE}; border-radius: 6px; padding: 7px 9px; background: {SURFACE};
    selection-background-color: {PRIMARY}; selection-color: {PRIMARY_TEXT};
}}
QLineEdit:focus, QAbstractSpinBox:focus {{ border: 2px solid {ACCENT}; padding: 6px 8px; }}

QPushButton {{
    border: 1px solid {LINE}; border-radius: 6px; padding: 7px 14px; background: {SIDEBAR};
}}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {PRESSED}; }}
QPushButton:focus {{ border: 2px solid {ACCENT}; padding: 6px 13px; }}
QPushButton:disabled {{ color: {FAINT}; background: {SIDEBAR}; }}
QPushButton#primary {{
    background: {PRIMARY}; color: {PRIMARY_TEXT}; border: 1px solid {ACCENT}; font-weight: 600;
    padding: 8px 22px;
}}
QPushButton#primary:hover {{ background: {PRIMARY_HOVER}; }}
QPushButton#primary:focus {{ border: 2px solid {INK}; padding: 7px 21px; }}
QPushButton#primary:disabled {{
    background: {PRIMARY_DISABLED}; border-color: {PRIMARY_DISABLED}; color: {FAINT};
}}
QPushButton#warning {{ background: {AMBER_SOFT}; border: 1px solid {AMBER_LINE}; }}
QPushButton#warning:hover {{ background: {AMBER_HOVER}; }}

QListWidget {{ border: none; background: {SIDEBAR}; outline: none; }}
QListWidget::item {{ border-bottom: 1px solid {LINE}; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; border-left: 3px solid {ACCENT}; }}
QListWidget::item:hover:!selected {{ background: {HOVER}; }}

QProgressBar {{
    border: none; border-radius: 4px; background: {TRACK}; height: 8px; max-height: 8px;
}}
QProgressBar::chunk {{ border-radius: 4px; background: {ACCENT}; }}
QProgressBar[kind="done"]::chunk {{ background: {GREEN}; }}
QProgressBar[kind="failed"]::chunk {{ background: {RED}; }}
QProgressBar[kind="cap"]::chunk {{ background: {AMBER_LINE}; }}

QStatusBar {{ color: {MUTED}; background: {SIDEBAR}; }}

QToolTip {{ background: {SIDEBAR}; color: {INK}; border: 1px solid {LINE}; padding: 4px; }}
QMessageBox, QInputDialog {{ background: {SURFACE}; }}
QScrollBar:vertical {{ background: {SURFACE}; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {FAINT}; }}
QScrollBar:horizontal {{ background: {SURFACE}; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {LINE}; border-radius: 5px; min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: {FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
"""


def set_kind(widget: QWidget, kind: str) -> None:
    """Set the ``kind`` property and re-apply the stylesheet so the color follows it."""
    if widget.property("kind") == kind:
        return
    widget.setProperty("kind", kind)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


LINK_VISITED = "#b392f0"


def apply_theme(application: QApplication) -> None:
    """Apply the dark stylesheet and the palette colors a stylesheet cannot reach (links)."""
    palette = application.palette()
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(LINK_VISITED))
    application.setPalette(palette)
    application.setStyleSheet(STYLESHEET)
