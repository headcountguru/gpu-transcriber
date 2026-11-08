"""Download and keep up to date the standalone ``yt-dlp.exe`` used for YouTube links."""

import http.client
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

YTDLP_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
YTDLP_EXE_NAME = "yt-dlp.exe"
STAMP_FILE_NAME = "yt-dlp-last-update.json"
UPDATE_INTERVAL_S = 24 * 3600.0
UPDATE_TIMEOUT_S = 180.0
DOWNLOAD_TIMEOUT_S = 60.0
DOWNLOAD_CHUNK_BYTES = 1 << 20
EXE_MAGIC = b"MZ"  # every Windows program starts with these two bytes


class YtDlpError(RuntimeError):
    """The YouTube helper could not be set up. The message is plain words for the user."""


def no_window_kwargs() -> dict[str, Any]:
    """Subprocess options that stop a console window flashing up on Windows."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def download_file(url: str, dest: Path) -> None:
    """Fetch ``url`` (always an https link chosen by this module) into ``dest``."""
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:  # noqa: S310
        announced = response.headers.get("Content-Length")
        written = 0
        with dest.open("wb") as out:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                out.write(chunk)
                written += len(chunk)
    if announced is not None and announced.isdigit() and written != int(announced):
        raise OSError(f"the download stopped early ({written} of {announced} bytes)")


def _looks_like_an_exe(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(len(EXE_MAGIC)) == EXE_MAGIC


def ensure_ytdlp(app_dir: Path, download: Callable[[str, Path], None] = download_file) -> Path:
    """Return the path to ``yt-dlp.exe`` in ``app_dir``, downloading it first if it is absent."""
    exe = app_dir / YTDLP_EXE_NAME
    if exe.is_file() and exe.stat().st_size > 0:
        return exe
    app_dir.mkdir(parents=True, exist_ok=True)
    part = exe.with_name(exe.name + ".part")
    try:
        download(YTDLP_URL, part)
        if not _looks_like_an_exe(part):
            raise OSError("the download is not a Windows program")
        os.replace(part, exe)
    except (OSError, http.client.HTTPException) as exc:
        logger.warning("yt-dlp download failed: %s", exc)
        part.unlink(missing_ok=True)
        raise YtDlpError(
            "Could not download the helper that fetches YouTube audio. "
            "Check your internet connection and try again."
        ) from exc
    logger.info("Downloaded yt-dlp to %s", exe)
    return exe


def _last_run(stamp_file: Path) -> float | None:
    try:
        value = json.loads(stamp_file.read_text(encoding="utf-8"))["last_run"]
        return float(value)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_stamp(stamp_file: Path, now: float) -> None:
    temp = stamp_file.with_name(stamp_file.name + ".tmp")
    try:
        temp.write_text(json.dumps({"last_run": now}), encoding="utf-8")
        os.replace(temp, stamp_file)
    except OSError as exc:
        logger.warning("Could not record the yt-dlp update time: %s", exc)


def update_ytdlp(
    exe: Path,
    run: Callable[..., Any] = subprocess.run,
    now: float | None = None,
    stamp_file: Path | None = None,
) -> bool:
    """Run ``yt-dlp -U`` at most once every 24 hours; True if it ran (or was tried).

    The attempt is recorded even if the update fails, so a broken network does not make every
    download pay for a fresh attempt. ``stamp_file`` defaults to a file next to the exe.
    """
    current = time.time() if now is None else now
    stamp = stamp_file if stamp_file is not None else exe.with_name(STAMP_FILE_NAME)
    last = _last_run(stamp)
    if last is not None and 0 <= current - last < UPDATE_INTERVAL_S:
        return False
    try:
        run(
            [str(exe), "-U"],
            capture_output=True,
            text=True,
            check=False,
            timeout=UPDATE_TIMEOUT_S,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("yt-dlp update could not run: %s", exc)
    _write_stamp(stamp, current)
    return True
