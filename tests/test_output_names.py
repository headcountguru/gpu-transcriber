"""Output files are named after the recording: ``<title>-transcript.txt`` and so on."""

from __future__ import annotations

import pytest

from app.jobs import output_file_name
from app.media_prep import strip_media_extension


@pytest.mark.parametrize(
    ("remote_name", "expected"),
    [
        ("transcript.txt", "Board meeting-transcript.txt"),
        ("transcript_timestamped.txt", "Board meeting-transcript_timestamped.txt"),
        ("transcript.srt", "Board meeting-transcript.srt"),
    ],
)
def test_each_output_file_starts_with_the_title_then_a_dash(
    remote_name: str, expected: str
) -> None:
    assert output_file_name("Board meeting", remote_name) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("City Council 09_08_2026.mp3", "City Council 09_08_2026"),
        ("Interview.MP4", "Interview"),
        ("podcast.m4a", "podcast"),
        ("clip.webm", "clip"),
        ("take 2.wav", "take 2"),
        ("saved as.opus", "saved as"),
        ("both.mp3.mp3", "both"),
        ("Version 1.2 recording", "Version 1.2 recording"),  # not a media extension
        ("Plan for 2026.", "Plan for 2026."),
        ("mp3", "mp3"),  # a name that is only the word, not an extension
        ("", ""),
    ],
)
def test_media_file_extensions_are_dropped_from_titles(title: str, expected: str) -> None:
    assert strip_media_extension(title) == expected
