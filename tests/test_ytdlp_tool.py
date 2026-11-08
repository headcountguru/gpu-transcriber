"""Tests for downloading and updating the standalone yt-dlp.exe."""

import http.client
import io
import json
import subprocess
from pathlib import Path

import pytest

from app.ytdlp_tool import (
    UPDATE_INTERVAL_S,
    YTDLP_URL,
    YtDlpError,
    ensure_ytdlp,
    update_ytdlp,
)

NOW = 1_000_000.0


class FakeRun:
    """Stands in for ``subprocess.run`` and records every command it was asked to run."""

    def __init__(self, returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self._returncode = returncode

    def __call__(self, args: list[str], **kwargs: object) -> "subprocess.CompletedProcess[str]":
        self.calls.append([str(a) for a in args])
        return subprocess.CompletedProcess(args, self._returncode, "", "")


def test_ensure_ytdlp_downloads_when_missing_and_leaves_no_part_file(tmp_path: Path) -> None:
    seen: list[str] = []

    def download(url: str, dest: Path) -> None:
        seen.append(url)
        dest.write_bytes(b"MZ-fake-exe")

    exe = ensure_ytdlp(tmp_path, download)

    assert exe == tmp_path / "yt-dlp.exe"
    assert exe.read_bytes() == b"MZ-fake-exe"
    assert seen == [YTDLP_URL]
    assert YTDLP_URL.startswith("https://github.com/yt-dlp/yt-dlp/releases/")
    assert list(tmp_path.glob("*.part")) == []


def test_ensure_ytdlp_does_not_download_when_present(tmp_path: Path) -> None:
    (tmp_path / "yt-dlp.exe").write_bytes(b"already here")

    def download(url: str, dest: Path) -> None:
        raise AssertionError("must not download")

    assert ensure_ytdlp(tmp_path, download) == tmp_path / "yt-dlp.exe"


def test_ensure_ytdlp_creates_missing_app_dir(tmp_path: Path) -> None:
    app_dir = tmp_path / "nested" / "bin"

    def download(url: str, dest: Path) -> None:
        dest.write_bytes(b"MZx")

    assert ensure_ytdlp(app_dir, download).is_file()


def test_ensure_ytdlp_rejects_a_download_that_is_not_an_exe(tmp_path: Path) -> None:
    def download(url: str, dest: Path) -> None:
        dest.write_bytes(b"<html>Rate limit exceeded</html>")

    with pytest.raises(YtDlpError):
        ensure_ytdlp(tmp_path, download)

    assert list(tmp_path.iterdir()) == []


def test_ensure_ytdlp_turns_http_client_errors_into_the_plain_error(tmp_path: Path) -> None:
    def download(url: str, dest: Path) -> None:
        dest.write_bytes(b"MZ-half")
        raise http.client.IncompleteRead(b"MZ-half", 100)

    with pytest.raises(YtDlpError) as caught:
        ensure_ytdlp(tmp_path, download)

    assert "internet" in str(caught.value).lower()
    assert list(tmp_path.iterdir()) == []


class FakeResponse(io.BytesIO):
    """A urlopen() result: a readable body plus the headers the server sent."""

    def __init__(self, body: bytes, content_length: int | None) -> None:
        super().__init__(body)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}


def test_download_file_rejects_a_body_shorter_than_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.ytdlp_tool.urllib.request.urlopen",
        lambda url, timeout: FakeResponse(b"MZ" + b"a" * 8, content_length=100),
    )

    with pytest.raises(YtDlpError):
        ensure_ytdlp(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_download_file_installs_a_complete_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"MZ" + b"a" * 98
    monkeypatch.setattr(
        "app.ytdlp_tool.urllib.request.urlopen",
        lambda url, timeout: FakeResponse(body, content_length=len(body)),
    )

    exe = ensure_ytdlp(tmp_path)

    assert exe.read_bytes() == body
    assert list(tmp_path.glob("*.part")) == []


def test_download_file_accepts_a_response_without_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.ytdlp_tool.urllib.request.urlopen",
        lambda url, timeout: FakeResponse(b"MZ-body", content_length=None),
    )

    assert ensure_ytdlp(tmp_path).read_bytes() == b"MZ-body"


def test_ensure_ytdlp_failed_download_gives_plain_error_and_cleans_up(tmp_path: Path) -> None:
    def download(url: str, dest: Path) -> None:
        dest.write_bytes(b"half")
        raise OSError("getaddrinfo failed")

    with pytest.raises(YtDlpError) as caught:
        ensure_ytdlp(tmp_path, download)

    assert "internet" in str(caught.value).lower()
    assert "getaddrinfo" not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_ensure_ytdlp_empty_download_is_an_error(tmp_path: Path) -> None:
    def download(url: str, dest: Path) -> None:
        dest.write_bytes(b"")

    with pytest.raises(YtDlpError):
        ensure_ytdlp(tmp_path, download)

    assert not (tmp_path / "yt-dlp.exe").exists()


def test_update_ytdlp_runs_at_most_once_a_day(tmp_path: Path) -> None:
    exe = tmp_path / "yt-dlp.exe"
    stamp = tmp_path / "stamp.json"
    run = FakeRun()

    first = update_ytdlp(exe, run=run, now=NOW, stamp_file=stamp)
    second = update_ytdlp(exe, run=run, now=NOW + 3600, stamp_file=stamp)

    assert first is True
    assert second is False
    assert run.calls == [[str(exe), "-U"]]
    assert json.loads(stamp.read_text(encoding="utf-8")) == {"last_run": NOW}


def test_update_ytdlp_runs_again_after_a_day(tmp_path: Path) -> None:
    exe = tmp_path / "yt-dlp.exe"
    stamp = tmp_path / "stamp.json"
    run = FakeRun()

    update_ytdlp(exe, run=run, now=NOW, stamp_file=stamp)
    again = update_ytdlp(exe, run=run, now=NOW + UPDATE_INTERVAL_S + 1, stamp_file=stamp)

    assert again is True
    assert len(run.calls) == 2


def test_update_ytdlp_treats_a_corrupt_stamp_as_never_run(tmp_path: Path) -> None:
    exe = tmp_path / "yt-dlp.exe"
    stamp = tmp_path / "stamp.json"
    stamp.write_text("not json at all", encoding="utf-8")
    run = FakeRun()

    assert update_ytdlp(exe, run=run, now=NOW, stamp_file=stamp) is True
    assert json.loads(stamp.read_text(encoding="utf-8")) == {"last_run": NOW}


def test_update_ytdlp_records_the_attempt_even_when_the_update_fails(tmp_path: Path) -> None:
    exe = tmp_path / "yt-dlp.exe"
    stamp = tmp_path / "stamp.json"

    def broken_run(args: list[str], **kwargs: object) -> "subprocess.CompletedProcess[str]":
        raise OSError("the exe is missing")

    assert update_ytdlp(exe, run=broken_run, now=NOW, stamp_file=stamp) is True
    assert update_ytdlp(exe, run=FakeRun(), now=NOW + 60, stamp_file=stamp) is False


def test_update_ytdlp_default_stamp_lives_next_to_the_exe(tmp_path: Path) -> None:
    exe = tmp_path / "yt-dlp.exe"

    update_ytdlp(exe, run=FakeRun(), now=NOW)

    assert [p.parent for p in tmp_path.iterdir()] == [tmp_path]
    assert len(list(tmp_path.iterdir())) == 1
