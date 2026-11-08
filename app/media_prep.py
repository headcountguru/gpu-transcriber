"""Prepare the user's recording on their own PC: compress it, measure it, or fetch a YouTube link.

Every failure is raised as :class:`MediaPrepError` with a message in plain words (no tracebacks,
no tool output). The orchestrator relies on that exception to stop before anything is rented.
"""

import logging
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from app.ytdlp_tool import no_window_kwargs, update_ytdlp

logger = logging.getLogger(__name__)

AUDIO_NAME = "audio.opus"
DOWNLOAD_NAME = "download.%(ext)s"
SAMPLE_RATE = "16000"
OPUS_BITRATE = "32k"
MAX_FOLDER_NAME_CHARS = 80
UNTITLED = "untitled"
FFMPEG_PROBE_TIMEOUT_S = 60.0
YTDLP_PROBE_TIMEOUT_S = 90.0
RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{n}" for n in range(1, 10)}
    | {f"LPT{n}" for n in range(1, 10)}
)

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_YOUTUBE_SHORT_HOST = "youtu.be"

MSG_NO_AUDIO = "This file has no audio to transcribe."
MSG_UNREADABLE = "Can't read this file. It may be damaged or not an audio or video file."
MSG_EMPTY = "This file is empty (it has no content)."
MSG_MISSING = "Can't find that file. It may have been moved or deleted."
MSG_CONVERT_FAILED = "Could not convert this file to audio."
MSG_PLAYLIST = "Playlists are not supported. Paste the link to a single video instead."
MSG_NOT_A_LINK = "That doesn't look like a web link. Paste the full link to a YouTube video."
MSG_UNAVAILABLE = (
    "That video is unavailable. It may be private, removed, or not offered in your country."
)
MSG_LIVE = "That video is a live stream, which can't be transcribed. Try again once it has ended."
MSG_SIGN_IN = (
    "YouTube is asking for a sign-in before it will share this video. "
    "Add your YouTube cookies in Settings and try again."
)
MSG_NETWORK = "Could not reach YouTube. Check your internet connection and try again."
MSG_NOT_A_VIDEO = "That link isn't a video that can be downloaded."
MSG_DOWNLOAD_FAILED = "Could not download the audio from that link."
MSG_DOWNLOAD_EMPTY = "The download finished but no audio came through."
MSG_TIMEOUT = "That took too long and was stopped. Please try again."
MSG_NO_LENGTH = (
    "Could not find out how long that video is. It may be a live stream, which can't be "
    "transcribed."
)

# (phrases in yt-dlp's message, plain-words answer), most specific first.
_YTDLP_PROBLEMS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("live event", "is live", "live stream"), MSG_LIVE),
    (
        (
            "unavailable",
            "private video",
            "this video is private",
            "has been removed",
            "no longer available",
            "account associated with this video has been terminated",
            "video does not exist",
        ),
        MSG_UNAVAILABLE,
    ),
    (("sign in to confirm", "confirm you're not a bot", "confirm your age"), MSG_SIGN_IN),
    (("unsupported url",), MSG_NOT_A_VIDEO),
    (
        ("unable to download webpage", "getaddrinfo", "timed out", "connection", "network"),
        MSG_NETWORK,
    ),
)
# (phrases in ffmpeg's message, plain-words answer).
_FFMPEG_PROBLEMS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "does not contain any stream",
            "matches no streams",
            "no audio",
            "stream map",
        ),
        MSG_NO_AUDIO,
    ),
    (("invalid data found", "moov atom not found", "could not find codec"), MSG_UNREADABLE),
    (("no such file", "does not exist"), MSG_MISSING),
)

Run = Callable[..., Any]


class MediaPrepError(RuntimeError):
    """Preparing the audio failed. The message is plain words for a non-technical user."""


@dataclass(frozen=True)
class PreparedAudio:
    """A small opus file ready to upload, plus what the user will recognise it by."""

    path: Path
    title: str
    duration_s: float
    size_bytes: int


def safe_folder_name(title: str) -> str:
    """Turn a title into a name that is safe as a Windows folder name (80 characters at most)."""
    cleaned = " ".join(_ILLEGAL_CHARS.sub("", title).split())
    cleaned = cleaned[:MAX_FOLDER_NAME_CHARS].rstrip(". ").lstrip()
    if not cleaned:
        return UNTITLED
    if cleaned.split(".")[0].rstrip().upper() in RESERVED_NAMES:
        cleaned = f"{cleaned}_"
    return cleaned


def compress_args(src: Path, dst: Path) -> list[str]:
    """ffmpeg arguments (after the exe) that make mono, 16 kHz, 32 kbps opus from ``src``."""
    return [
        "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", SAMPLE_RATE,
        "-c:a", "libopus", "-b:a", OPUS_BITRATE, str(dst),
    ]  # fmt: skip


def _classify(text: str, table: tuple[tuple[tuple[str, ...], str], ...], default: str) -> str:
    lowered = text.lower()
    for phrases, answer in table:
        if any(phrase in lowered for phrase in phrases):
            return answer
    return default


def _run_tool(
    run: Run, command: list[str], timeout: float | None
) -> "subprocess.CompletedProcess[str]":
    """Run a command with captured text output and no console window."""
    try:
        return run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
            **no_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("%s timed out", Path(command[0]).name)
        raise MediaPrepError(MSG_TIMEOUT) from exc
    except OSError as exc:
        logger.warning("%s could not start: %s", Path(command[0]).name, exc)
        raise MediaPrepError(
            "A helper program needed to prepare the audio could not be started."
        ) from exc


def _check_source(source: Path) -> None:
    if not source.is_file():
        raise MediaPrepError(MSG_MISSING)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise MediaPrepError(MSG_UNREADABLE) from exc
    if size == 0:
        raise MediaPrepError(MSG_EMPTY)


def _parse_duration(stderr: str) -> float | None:
    match = _DURATION_RE.search(stderr)
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _probe_media(path: Path, ffmpeg_exe: str, run: Run) -> float:
    """Length in seconds from ffmpeg's header; raises if there is no readable audio."""
    result = _run_tool(run, [ffmpeg_exe, "-hide_banner", "-i", str(path)], FFMPEG_PROBE_TIMEOUT_S)
    header = result.stderr or ""
    duration = _parse_duration(header)
    if duration is None:
        raise MediaPrepError(_classify(header, _FFMPEG_PROBLEMS, MSG_UNREADABLE))
    if " Audio:" not in header:
        raise MediaPrepError(MSG_NO_AUDIO)
    if duration <= 0:
        raise MediaPrepError(MSG_EMPTY)
    return duration


def _compress(source: Path, work_dir: Path, ffmpeg_exe: str, title: str, run: Run) -> PreparedAudio:
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / AUDIO_NAME
    result = _run_tool(run, [ffmpeg_exe, *compress_args(source, target)], None)
    if result.returncode != 0 or not target.is_file():
        target.unlink(missing_ok=True)
        logger.warning("ffmpeg conversion failed (exit %s)", result.returncode)
        raise MediaPrepError(_classify(result.stderr or "", _FFMPEG_PROBLEMS, MSG_CONVERT_FAILED))
    try:
        duration = _probe_media(target, ffmpeg_exe, run)
    except MediaPrepError:
        target.unlink(missing_ok=True)
        raise
    return PreparedAudio(
        path=target, title=title, duration_s=duration, size_bytes=target.stat().st_size
    )


def prepare_file(
    source: Path, work_dir: Path, ffmpeg_exe: str, *, run: Run = subprocess.run
) -> PreparedAudio:
    """Compress a local audio or video file to ``work_dir/audio.opus``."""
    _check_source(source)
    _probe_media(source, ffmpeg_exe, run)
    return _compress(source, work_dir, ffmpeg_exe, source.stem, run)


def _check_youtube_url(url: str) -> str:
    text = url.strip()
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise MediaPrepError(MSG_NOT_A_LINK)
    query = parse_qs(parsed.query)
    is_short_video_link = parsed.netloc.lower().endswith(_YOUTUBE_SHORT_HOST) and bool(
        parsed.path.strip("/")
    )
    if "list" in query and "v" not in query and not is_short_video_link:
        raise MediaPrepError(MSG_PLAYLIST)
    return text


def _ytdlp_base(ytdlp_exe: Path, ffmpeg_exe: str, cookies_file: Path | None) -> list[str]:
    command = [str(ytdlp_exe), "--no-playlist", "--no-warnings", "--ffmpeg-location", ffmpeg_exe]
    if cookies_file is not None:
        command += ["--cookies", str(cookies_file)]
    return command


def _run_ytdlp(command: list[str], ytdlp_exe: Path, timeout: float | None, run: Run) -> str:
    """Run yt-dlp, updating it once and retrying once if it fails; return its stdout."""
    result = _run_tool(run, command, timeout)
    if result.returncode != 0 and update_ytdlp(ytdlp_exe, run=run):
        logger.info("yt-dlp failed; retrying after an update")
        result = _run_tool(run, command, timeout)
    if result.returncode != 0:
        logger.warning("yt-dlp failed (exit %s)", result.returncode)
        raise MediaPrepError(_classify(result.stderr or "", _YTDLP_PROBLEMS, MSG_DOWNLOAD_FAILED))
    return result.stdout or ""


def prepare_youtube(
    url: str,
    work_dir: Path,
    ffmpeg_exe: str,
    ytdlp_exe: Path,
    cookies_file: Path | None = None,
    *,
    run: Run = subprocess.run,
) -> PreparedAudio:
    """Download a YouTube video's audio and compress it to ``work_dir/audio.opus``."""
    link = _check_youtube_url(url)
    work_dir.mkdir(parents=True, exist_ok=True)
    command = _ytdlp_base(ytdlp_exe, ffmpeg_exe, cookies_file) + [
        "-f", "bestaudio", "--no-simulate",
        "--print", "before_dl:%(title)s", "--print", "after_move:filepath",
        "-o", str(work_dir / DOWNLOAD_NAME), link,
    ]  # fmt: skip
    lines = [line.strip() for line in _run_ytdlp(command, ytdlp_exe, None, run).splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < 2:
        raise MediaPrepError(MSG_DOWNLOAD_EMPTY)
    title, downloaded = lines[0], Path(lines[-1])
    try:
        _check_source(downloaded)
    except MediaPrepError as exc:
        raise MediaPrepError(MSG_DOWNLOAD_EMPTY) from exc
    try:
        return _compress(downloaded, work_dir, ffmpeg_exe, title, run)
    finally:
        downloaded.unlink(missing_ok=True)


def probe_duration_s(
    source: str,
    is_youtube: bool,
    ffmpeg_exe: str,
    ytdlp_exe: Path,
    cookies_file: Path | None = None,
    *,
    run: Run = subprocess.run,
) -> float:
    """Length of a recording in seconds, without downloading or converting it."""
    if not is_youtube:
        path = Path(source)
        _check_source(path)
        return _probe_media(path, ffmpeg_exe, run)
    link = _check_youtube_url(source)
    command = _ytdlp_base(ytdlp_exe, ffmpeg_exe, cookies_file) + [
        "--skip-download", "--print", "duration", link,
    ]  # fmt: skip
    output = _run_ytdlp(command, ytdlp_exe, YTDLP_PROBE_TIMEOUT_S, run)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    try:
        seconds = float(lines[-1])
    except (IndexError, ValueError) as exc:
        raise MediaPrepError(MSG_NO_LENGTH) from exc
    if seconds <= 0:
        raise MediaPrepError(MSG_NO_LENGTH)
    return seconds
