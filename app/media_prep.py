"""Prepare the user's recording on their own PC: compress it, measure it, or fetch a YouTube link.

Every failure is raised as :class:`MediaPrepError` with a message in plain words (no tracebacks,
no tool output). The orchestrator relies on that exception to stop before anything is rented.
"""

import logging
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, parse_qs, urlparse

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
DOWNLOAD_TIMEOUT_S = 6 * 3600.0  # a very long video on a slow link; finite so nothing hangs forever
RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{n}" for n in range(1, 10)}
    | {f"LPT{n}" for n in range(1, 10)}
)

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_NO_DURATION_RE = re.compile(r"Duration:\s*N/A")
_YOUTUBE_SHORT_HOST = "youtu.be"
_YOUTUBE_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"})
_VIDEO_PATH_KINDS = frozenset({"shorts", "live", "embed"})
_PARTIAL_SUFFIXES = frozenset({".part", ".ytdl", ".temp"})
MAX_DOWNLOADS_REACHED = 101  # yt-dlp's exit code once --max-downloads is satisfied
_DOWNLOAD_OK_CODES = frozenset({0, MAX_DOWNLOADS_REACHED})

MSG_NO_AUDIO = "This file has no audio to transcribe."
MSG_UNREADABLE = "Can't read this file. It may be damaged or not an audio or video file."
MSG_EMPTY = "This file is empty (it has no content)."
MSG_MISSING = "Can't find that file. It may have been moved or deleted."
MSG_CONVERT_FAILED = "Could not convert this file to audio."
MSG_WORKING_FOLDER = (
    "Could not save the converted audio in the app's working folder. Your file is fine. "
    "Check that the app's data folder can be written to, then try again."
)
MSG_PLAYLIST = "Playlists are not supported. Paste the link to a single video instead."
MSG_NOT_A_LINK = "That doesn't look like a web link. Paste the full link to a YouTube video."
MSG_NOT_SINGLE_VIDEO = (
    "That link isn't a single video from YouTube. Paste the link to one video "
    "(not a channel, a playlist or another website)."
)
MSG_UNAVAILABLE = (
    "That video is unavailable. It may be private, removed, or not offered in your country."
)
MSG_LIVE = "That video is a live stream, which can't be transcribed. Try again once it has ended."
MSG_SIGN_IN = (
    "YouTube is asking for a sign-in before it will share this video (for example an "
    "age-restricted or members-only video), and this app can't sign in to YouTube. "
    "Download the video another way, then choose the audio or video file instead of the link."
)
MSG_NETWORK = "Could not reach YouTube. Check your internet connection and try again."
MSG_NOT_A_VIDEO = "That link isn't a video that can be downloaded."
MSG_DOWNLOAD_FAILED = "Could not download the audio from that link."
MSG_DOWNLOAD_EMPTY = "The download finished but no audio came through."
MSG_TIMEOUT = "That took too long and was stopped. Please try again."
MSG_LENGTH_UNKNOWN = (
    "We can't tell how long this file is, so there is no estimate. You can still start the job."
)
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
    # The input file was already checked to exist, so a "no such file" from ffmpeg is about the
    # place it writes the converted audio to (seen live under the Microsoft Store Python).
    (("no such file", "does not exist"), MSG_WORKING_FOLDER),
)

Run = Callable[..., Any]


class MediaPrepError(RuntimeError):
    """Preparing the audio failed. The message is plain words for a non-technical user."""


class UnknownLengthError(MediaPrepError):
    """The file is readable but its header has no length, so no estimate can be made."""


@dataclass(frozen=True)
class PreparedAudio:
    """A small opus file ready to upload, plus what the user will recognise it by."""

    path: Path
    title: str
    duration_s: float
    size_bytes: int


_MEDIA_EXTENSION = re.compile(
    r"\.(mp3|mp4|m4a|m4b|aac|wav|wave|flac|ogg|oga|opus|wma|webm|mkv|mov|avi|wmv|flv|mpg|mpeg|3gp)$",
    re.IGNORECASE,
)


def strip_media_extension(title: str) -> str:
    """Remove trailing media file extensions (``.mp3``, ``.MP4``...) from a title or file name."""
    stripped = title
    while True:
        shorter = _MEDIA_EXTENSION.sub("", stripped)
        if shorter == stripped or not shorter:
            return stripped
        stripped = shorter


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
    run: Run,
    command: list[str],
    timeout: float | None,
    env: dict[str, str] | None = None,
) -> "subprocess.CompletedProcess[str]":
    """Run a command with captured text output and no console window."""
    extra: dict[str, Any] = {} if env is None else {"env": env}
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
            **extra,
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


def _probe_media(path: Path, ffmpeg_exe: str, run: Run) -> float | None:
    """Length in seconds from ffmpeg's header; raises if there is no readable audio.

    None means the header says ``Duration: N/A`` (some files, e.g. browser-recorded .webm,
    carry no length): the audio is readable, only its length is not known up front.
    """
    result = _run_tool(run, [ffmpeg_exe, "-hide_banner", "-i", str(path)], FFMPEG_PROBE_TIMEOUT_S)
    header = result.stderr or ""
    duration = _parse_duration(header)
    if duration is None and _NO_DURATION_RE.search(header) is None:
        raise MediaPrepError(_classify(header, _FFMPEG_PROBLEMS, MSG_UNREADABLE))
    if " Audio:" not in header:
        raise MediaPrepError(MSG_NO_AUDIO)
    if duration is not None and duration <= 0:
        raise MediaPrepError(MSG_EMPTY)
    return duration


def _probe_known_length(path: Path, ffmpeg_exe: str, run: Run) -> float:
    duration = _probe_media(path, ffmpeg_exe, run)
    if duration is None:
        raise UnknownLengthError(MSG_LENGTH_UNKNOWN)
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
        duration = _probe_known_length(target, ffmpeg_exe, run)  # the encoded file has one
    except MediaPrepError:
        target.unlink(missing_ok=True)
        raise
    return PreparedAudio(
        path=target, title=title, duration_s=duration, size_bytes=target.stat().st_size
    )


def prepare_file(
    source: Path, work_dir: Path, ffmpeg_exe: str, *, run: Run = subprocess.run
) -> PreparedAudio:
    """Compress a local audio or video file to ``work_dir/audio.opus``.

    A source without a length in its header is fine: the length is measured on the output.
    """
    _check_source(source)
    _probe_media(source, ffmpeg_exe, run)
    return _compress(source, work_dir, ffmpeg_exe, source.stem, run)


def _is_single_video(parsed: ParseResult, host: str) -> bool:
    """True for watch?v=, youtu.be/<id>, shorts/, live/ and embed/ links on real YouTube hosts."""
    parts = [part for part in parsed.path.split("/") if part]
    if host == _YOUTUBE_SHORT_HOST:
        return len(parts) == 1
    if host not in _YOUTUBE_HOSTS:
        return False
    if parts == ["watch"]:
        return bool(parse_qs(parsed.query).get("v"))
    return len(parts) == 2 and parts[0] in _VIDEO_PATH_KINDS


def _check_youtube_url(url: str) -> str:
    text = url.strip()
    try:
        parsed = urlparse(text)
        host = (parsed.hostname or "").lower()
    except ValueError as exc:
        raise MediaPrepError(MSG_NOT_SINGLE_VIDEO) from exc
    if parsed.scheme not in ("http", "https") or not host:
        raise MediaPrepError(MSG_NOT_A_LINK)
    if _is_single_video(parsed, host):
        return text
    if (host in _YOUTUBE_HOSTS or host == _YOUTUBE_SHORT_HOST) and "list" in parse_qs(parsed.query):
        raise MediaPrepError(MSG_PLAYLIST)
    raise MediaPrepError(MSG_NOT_SINGLE_VIDEO)


def check_youtube_url(url: str) -> str:
    """Return the trimmed link if it is one YouTube video; else raise a plain MediaPrepError.

    Checks the text only (no network), so the window can enable Play as the user types.
    """
    return _check_youtube_url(url)


def _ytdlp_base(ytdlp_exe: Path, ffmpeg_exe: str, cookies_file: Path | None) -> list[str]:
    command = [str(ytdlp_exe), "--no-playlist", "--no-warnings", "--ffmpeg-location", ffmpeg_exe]
    if cookies_file is not None:
        command += ["--cookies", str(cookies_file)]
    return command


def _ytdlp_env() -> dict[str, str]:
    """The environment for yt-dlp, forcing UTF-8 on its piped output."""
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def _run_ytdlp(
    command: list[str],
    ytdlp_exe: Path,
    timeout: float | None,
    run: Run,
    ok_codes: frozenset[int] = frozenset({0}),
) -> str:
    """Run yt-dlp, updating it once and retrying once if it fails; return its stdout."""
    result = _run_tool(run, command, timeout, _ytdlp_env())
    if result.returncode not in ok_codes and update_ytdlp(ytdlp_exe, run=run):
        logger.info("yt-dlp failed; retrying after an update")
        result = _run_tool(run, command, timeout, _ytdlp_env())
    if result.returncode not in ok_codes:
        logger.warning("yt-dlp failed (exit %s)", result.returncode)
        raise MediaPrepError(_classify(result.stderr or "", _YTDLP_PROBLEMS, MSG_DOWNLOAD_FAILED))
    return result.stdout or ""


def _download_files(work_dir: Path) -> list[Path]:
    return [p for p in work_dir.glob("download.*") if p.is_file()]


def _clear_downloads(work_dir: Path) -> None:
    """Delete every ``download.*`` file (stale or leftover) from ``work_dir``."""
    for path in _download_files(work_dir):
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Could not remove %s: %s", path.name, exc)


def _find_download(work_dir: Path) -> Path | None:
    """The newest finished ``download.*`` file (``.part``/``.ytdl`` leftovers are skipped)."""
    finished = [p for p in _download_files(work_dir) if p.suffix.lower() not in _PARTIAL_SUFFIXES]
    return max(finished, key=lambda p: p.stat().st_mtime, default=None)


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
    _clear_downloads(work_dir)
    command = _ytdlp_base(ytdlp_exe, ffmpeg_exe, cookies_file) + [
        "-f", "bestaudio", "--no-simulate", "--match-filter", "!is_live", "--max-downloads", "1",
        "--print", "before_dl:%(title)s", "--print", "after_move:filepath",
        "-o", str(work_dir / DOWNLOAD_NAME), link,
    ]  # fmt: skip
    output = _run_ytdlp(command, ytdlp_exe, DOWNLOAD_TIMEOUT_S, run, _DOWNLOAD_OK_CODES)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    # The printed path is not trusted (piped output may be mangled for non-ASCII folders).
    downloaded = _find_download(work_dir)
    if downloaded is None:
        raise MediaPrepError(MSG_DOWNLOAD_EMPTY if lines else MSG_LIVE)
    try:
        _check_source(downloaded)
    except MediaPrepError as exc:
        raise MediaPrepError(MSG_DOWNLOAD_EMPTY) from exc
    title = lines[0] if len(lines) >= 2 else UNTITLED
    try:
        return _compress(downloaded, work_dir, ffmpeg_exe, title, run)
    finally:
        _clear_downloads(work_dir)


def probe_duration_s(
    source: str,
    is_youtube: bool,
    ffmpeg_exe: str,
    ytdlp_exe: Path,
    cookies_file: Path | None = None,
    *,
    run: Run = subprocess.run,
) -> float:
    """Length of a recording in seconds, without downloading or converting it.

    Raises ``UnknownLengthError`` (a ``MediaPrepError``) for a readable file whose header has
    no length: there is then no estimate, but the file can still be transcribed.
    """
    if not is_youtube:
        path = Path(source)
        _check_source(path)
        return _probe_known_length(path, ffmpeg_exe, run)
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
