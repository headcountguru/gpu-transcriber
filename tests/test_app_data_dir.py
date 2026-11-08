"""Where the app keeps its files.

The Microsoft Store build of Python silently redirects writes under ``%APPDATA%`` into a private
per-app copy that other programs (ffmpeg, yt-dlp) cannot see, so the first live run failed with
"Can't find that file" although the audio was fine. Under that Python the app must keep its data
outside ``AppData``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import settings

STORE_PREFIX = r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.10_x64__abc"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(settings.DATA_DIR_ENV, raising=False)


def test_normal_python_uses_appdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setattr(settings.sys, "platform", "win32")
    monkeypatch.setattr(settings.sys, "base_prefix", r"C:\Python312")

    path = settings.app_data_dir()

    assert path == tmp_path / "Roaming" / settings.APP_DATA_DIR_NAME
    assert path.is_dir()


def test_store_python_keeps_data_outside_appdata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setattr(settings.Path, "home", staticmethod(lambda: tmp_path / "home"))
    monkeypatch.setattr(settings.sys, "platform", "win32")
    monkeypatch.setattr(settings.sys, "base_prefix", STORE_PREFIX)

    path = settings.app_data_dir()

    assert path == tmp_path / "home" / settings.APP_DATA_DIR_NAME
    assert path.is_dir()
    assert not (tmp_path / "Roaming").exists()


def test_environment_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    chosen = tmp_path / "somewhere" / "else"
    monkeypatch.setenv(settings.DATA_DIR_ENV, str(chosen))
    monkeypatch.setattr(settings.sys, "base_prefix", STORE_PREFIX)

    path = settings.app_data_dir()

    assert path == chosen
    assert path.is_dir()


def test_store_detection_is_off_for_other_platforms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setattr(settings.sys, "platform", "linux")
    monkeypatch.setattr(settings.sys, "base_prefix", STORE_PREFIX)

    assert settings.app_data_dir() == tmp_path / "Roaming" / settings.APP_DATA_DIR_NAME
