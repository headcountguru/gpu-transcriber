"""The window's dark theme must stay readable (WCAG contrast) and have no stray light colors."""

from __future__ import annotations

import re

import pytest

from app.gui import theme

MIN_BODY = 4.5  # WCAG AA for normal text
MIN_LARGE = 3.0  # WCAG AA for large or secondary text


def _channel(value: int) -> float:
    c = value / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(color: str) -> float:
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(foreground: str, background: str) -> float:
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize("background", [theme.SURFACE, theme.SIDEBAR])
@pytest.mark.parametrize(
    "foreground", [theme.INK, theme.MUTED, theme.ACCENT, theme.GREEN, theme.RED, theme.AMBER]
)
def test_text_colors_are_readable_on_both_backgrounds(foreground: str, background: str) -> None:
    assert contrast(foreground, background) >= MIN_BODY


@pytest.mark.parametrize("background", [theme.SURFACE, theme.SIDEBAR])
def test_faint_notes_are_still_legible(background: str) -> None:
    assert contrast(theme.FAINT, background) >= MIN_LARGE


def test_primary_button_text_is_readable() -> None:
    assert contrast(theme.PRIMARY_TEXT, theme.PRIMARY) >= MIN_BODY
    assert contrast(theme.PRIMARY_TEXT, theme.PRIMARY_HOVER) >= MIN_BODY


@pytest.mark.parametrize(
    ("foreground", "background"),
    [
        (theme.RED, theme.RED_SOFT),
        (theme.INK, theme.AMBER_SOFT),
        (theme.INK, theme.ACCENT_SOFT),
        (theme.INK, theme.HOVER),
        (theme.INK, theme.PRESSED),
    ],
)
def test_text_on_tinted_backgrounds_is_readable(foreground: str, background: str) -> None:
    assert contrast(foreground, background) >= MIN_BODY


def test_the_page_is_dark() -> None:
    assert _luminance(theme.SURFACE) < 0.05
    assert _luminance(theme.SIDEBAR) < 0.05


def test_stylesheet_has_no_light_leftovers() -> None:
    for color in re.findall(r"#[0-9a-fA-F]{6}", theme.STYLESHEET):
        assert _luminance(color) < 0.5 or color in {
            theme.INK, theme.MUTED, theme.FAINT, theme.ACCENT, theme.GREEN, theme.RED,
            theme.AMBER, theme.PRIMARY_TEXT,
        }, f"{color} looks like a leftover light color"  # fmt: skip
    assert "white" not in theme.STYLESHEET.lower()


def test_apply_theme_makes_links_readable_on_the_dark_page(qapp: object) -> None:
    # Rich-text links (setup dialog) take their color from the palette, not the stylesheet; the
    # default dark blue is unreadable on a near-black page.
    from PySide6.QtGui import QColor, QPalette

    theme.apply_theme(qapp)  # type: ignore[arg-type]
    palette = qapp.palette()  # type: ignore[attr-defined]

    assert palette.color(QPalette.ColorRole.Link) == QColor(theme.ACCENT)
    assert contrast(palette.color(QPalette.ColorRole.LinkVisited).name(), theme.SURFACE) >= MIN_BODY
    assert qapp.styleSheet() == theme.STYLESHEET  # type: ignore[attr-defined]
