"""User-facing words must describe features that exist and say honestly what closing costs."""

from pathlib import Path

from app.gui.main_window import CLOSE_TEXT
from app.media_prep import MSG_SIGN_IN
from app.orchestrator import MSG_NO_TOKEN
from common.protocol import GRACE_AFTER_FINISH_S

README = Path(__file__).resolve().parents[1] / "README.md"
KEYS_BUTTON = "Accounts and keys…"


def test_sign_in_message_offers_a_real_way_out() -> None:
    assert "Settings" not in MSG_SIGN_IN
    assert "cookies" not in MSG_SIGN_IN.lower()
    assert "sign-in" in MSG_SIGN_IN
    assert "choose the audio or video file" in MSG_SIGN_IN


def test_missing_token_message_names_the_real_button() -> None:
    assert "Settings" not in MSG_NO_TOKEN
    assert KEYS_BUTTON in MSG_NO_TOKEN


def test_close_dialog_says_the_machine_keeps_charging_and_the_cap_stops() -> None:
    text = CLOSE_TEXT.lower()
    assert "charging" in text
    assert "spending cap" in text and "only while the app is open" in text
    assert f"{GRACE_AFTER_FINISH_S / 3600:.0f} hours" in text
    assert "an hour" in text and "$" in text
    assert "reopen the app" in text


def test_readme_close_section_is_honest() -> None:
    text = README.read_text(encoding="utf-8")
    section = text.split("**If you close the app during a job.**", 1)[1].split("\n\n", 1)[0]
    assert "charging" in section
    assert "only while the app is open" in section
    assert f"{GRACE_AFTER_FINISH_S / 3600:.0f} hours" in section
    assert "Reopen the app" in section or "reopen the app" in section


def test_readme_troubleshooting_covers_youtube_sign_in() -> None:
    text = README.read_text(encoding="utf-8")
    troubleshooting = text.split("## Troubleshooting", 1)[1]
    assert "asking for a sign-in" in troubleshooting
    assert "cookies" not in text.lower()
