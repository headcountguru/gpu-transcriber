"""Files whose header has no length (``Duration: N/A``, e.g. browser-recorded .webm)."""

import re
import subprocess
from pathlib import Path

import pytest

from app.gui.new_job_panel import MSG_LENGTH_UNKNOWN, NewJobPanel
from app.media_prep import MediaPrepError, UnknownLengthError, prepare_file, probe_duration_s
from tests.test_media_prep import FFMPEG, make_silent_video, make_wav

WAIT_MS = 5000


class HeaderWithoutLength:
    """``subprocess.run`` that reports ``Duration: N/A`` whenever ffmpeg reads ``source``."""

    def __init__(self, source: Path) -> None:
        self.source = str(source)

    def __call__(self, args: list[str], **kwargs: object) -> "subprocess.CompletedProcess[str]":
        result = subprocess.run(args, **kwargs)  # type: ignore[call-overload]
        command = [str(a) for a in args]
        if "-i" in command and command[command.index("-i") + 1] == self.source:
            stderr = re.sub(r"Duration: [^,]+", "Duration: N/A", result.stderr or "")
            return subprocess.CompletedProcess(args, result.returncode, result.stdout, stderr)
        return result


def test_prepare_file_accepts_a_source_without_a_length(tmp_path: Path) -> None:
    source = make_wav(tmp_path / "browser.wav", seconds=2.0)

    prepared = prepare_file(source, tmp_path / "work", FFMPEG, run=HeaderWithoutLength(source))

    assert prepared.path.is_file()
    assert prepared.duration_s == pytest.approx(2.0, abs=0.3)  # measured on the encoded output


def test_prepare_file_still_rejects_a_lengthless_file_without_audio(tmp_path: Path) -> None:
    source = make_silent_video(tmp_path / "silent.mp4")

    with pytest.raises(MediaPrepError, match="no audio"):
        prepare_file(source, tmp_path / "work", FFMPEG, run=HeaderWithoutLength(source))


def test_probe_duration_says_plainly_that_the_length_is_unknown(tmp_path: Path) -> None:
    source = make_wav(tmp_path / "browser.wav")

    with pytest.raises(UnknownLengthError) as err:
        probe_duration_s(
            str(source), False, FFMPEG, tmp_path / "yt-dlp.exe", run=HeaderWithoutLength(source)
        )

    assert "can't tell how long this file is" in str(err.value)


def test_panel_shows_length_unknown_and_still_allows_play(qtbot: object, tmp_path: Path) -> None:
    source = make_wav(tmp_path / "browser.wav")

    def probe(_source: str, _is_youtube: bool) -> float:
        raise UnknownLengthError("We can't tell how long this file is.")

    panel = NewJobPanel(probe=probe, debounce_ms=0)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.source_edit.setText(str(source))

    qtbot.waitUntil(lambda: panel.estimate_label.text() == MSG_LENGTH_UNKNOWN, timeout=WAIT_MS)  # type: ignore[attr-defined]
    qtbot.waitUntil(panel.play_button.isEnabled, timeout=WAIT_MS)  # type: ignore[attr-defined]
    assert not panel.error_label.text()
    qtbot.waitUntil(lambda: not panel.is_probing(), timeout=WAIT_MS)  # type: ignore[attr-defined]
