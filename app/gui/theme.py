"""Colors, fonts and the one stylesheet for the window.

Colors are semantic: red only for failures, amber for caps and alarms, green for done, blue for
the one primary action and for what is in progress. Widgets pick a color by setting the
``kind`` property (see ``set_kind``) instead of inline styles.
"""

from PySide6.QtWidgets import QWidget

BASE_POINT_SIZE = 10
FONT_FAMILY = "Segoe UI"

INK = "#1f2328"
MUTED = "#59636e"
FAINT = "#8c959f"
LINE = "#d0d7de"
SIDEBAR = "#f6f8fa"
SURFACE = "#ffffff"
ACCENT = "#0969da"
ACCENT_DARK = "#0550ae"
ACCENT_SOFT = "#ddf4ff"
GREEN = "#1a7f37"
RED = "#cf222e"
RED_SOFT = "#ffebe9"
AMBER = "#9a6700"
AMBER_SOFT = "#fff8c5"
AMBER_LINE = "#d4a72c"

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

QLineEdit {{
    border: 1px solid {LINE}; border-radius: 6px; padding: 7px 9px; background: {SURFACE};
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{ border: 2px solid {ACCENT}; padding: 6px 8px; }}

QPushButton {{
    border: 1px solid {LINE}; border-radius: 6px; padding: 7px 14px; background: {SIDEBAR};
}}
QPushButton:hover {{ background: #eef1f4; }}
QPushButton:pressed {{ background: #e1e5e9; }}
QPushButton:focus {{ border: 2px solid {ACCENT}; padding: 6px 13px; }}
QPushButton:disabled {{ color: {FAINT}; background: {SIDEBAR}; }}
QPushButton#primary {{
    background: {ACCENT}; color: white; border: 1px solid {ACCENT_DARK}; font-weight: 600;
    padding: 8px 22px;
}}
QPushButton#primary:hover {{ background: {ACCENT_DARK}; }}
QPushButton#primary:focus {{ border: 2px solid {INK}; padding: 7px 21px; }}
QPushButton#primary:disabled {{ background: #9ec5f4; border-color: #9ec5f4; color: white; }}
QPushButton#warning {{ background: {AMBER_SOFT}; border: 1px solid {AMBER_LINE}; }}
QPushButton#warning:hover {{ background: #fff1a8; }}

QListWidget {{ border: none; background: {SIDEBAR}; outline: none; }}
QListWidget::item {{ border-bottom: 1px solid {LINE}; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; border-left: 3px solid {ACCENT}; }}
QListWidget::item:hover:!selected {{ background: #eef1f4; }}

QProgressBar {{
    border: none; border-radius: 4px; background: #eaeef2; height: 8px; max-height: 8px;
}}
QProgressBar::chunk {{ border-radius: 4px; background: {ACCENT}; }}
QProgressBar[kind="done"]::chunk {{ background: {GREEN}; }}
QProgressBar[kind="failed"]::chunk {{ background: {RED}; }}
QProgressBar[kind="cap"]::chunk {{ background: {AMBER_LINE}; }}

QStatusBar {{ color: {MUTED}; }}
"""


def set_kind(widget: QWidget, kind: str) -> None:
    """Set the ``kind`` property and re-apply the stylesheet so the color follows it."""
    if widget.property("kind") == kind:
        return
    widget.setProperty("kind", kind)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
