"""Tests for the wiring in ``app.main`` (no network, no window shown, nothing launched)."""

from pathlib import Path

import pytest

from app import main as app_main
from app.credentials import VAST_API_KEY, CredentialStore
from app.jobs import JobStore
from app.media_prep import MediaPrepError, PreparedAudio
from app.vast_client import Instance
from app.ytdlp_tool import YtDlpError
from tests.gui_fakes import FakeBackend, make_job


def prepared(path: Path) -> PreparedAudio:
    return PreparedAudio(path=path, title="t", duration_s=1.0, size_bytes=1)


def test_selftest_exits_zero() -> None:
    assert app_main.main(["--selftest"]) == 0


def test_prepare_youtube_job_goes_to_its_audio_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path / "jobs")
    calls: list[tuple[object, ...]] = []

    def fake_youtube(url: str, work_dir: Path, ffmpeg: str, ytdlp: Path) -> PreparedAudio:
        calls.append((url, work_dir, ffmpeg, ytdlp))
        return prepared(work_dir / "audio.opus")

    monkeypatch.setattr(app_main, "prepare_youtube", fake_youtube)
    prepare = app_main.make_prepare(store, lambda: "ffmpeg.exe", lambda: Path("yt-dlp.exe"))
    job = make_job("y1", source="https://youtu.be/x", is_youtube=True)
    prepare(job)
    assert calls == [
        ("https://youtu.be/x", store.job_dir("y1") / "audio", "ffmpeg.exe", Path("yt-dlp.exe"))
    ]


def test_prepare_file_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JobStore(tmp_path / "jobs")
    calls: list[tuple[object, ...]] = []

    def fake_file(source: Path, work_dir: Path, ffmpeg: str) -> PreparedAudio:
        calls.append((source, work_dir, ffmpeg))
        return prepared(work_dir / "audio.opus")

    monkeypatch.setattr(app_main, "prepare_file", fake_file)
    prepare = app_main.make_prepare(store, lambda: "ff", lambda: Path("never-used"))
    prepare(make_job("f1", source=r"C:\rec\talk.mp3", is_youtube=False))
    assert calls == [(Path(r"C:\rec\talk.mp3"), store.job_dir("f1") / "audio", "ff")]


def test_prepare_turns_missing_ytdlp_into_plain_media_error(tmp_path: Path) -> None:
    def no_ytdlp() -> Path:
        raise YtDlpError("Could not download the helper that fetches YouTube audio.")

    prepare = app_main.make_prepare(JobStore(tmp_path), lambda: "ff", no_ytdlp)
    with pytest.raises(MediaPrepError, match="helper that fetches"):
        prepare(make_job("y2", is_youtube=True))


def test_session_factory_refuses_instance_without_address() -> None:
    factory = app_main.make_session_factory(Path("key"))
    instance = Instance(
        id=1, status="running", ssh_host=None, ssh_port=None, dph_total=0.2, label="x"
    )
    with pytest.raises(ConnectionError):
        factory(instance)


def test_session_factory_connects_with_the_app_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_connect(host: str, port: int, key: str, **kwargs: object) -> str:
        seen.update(host=host, port=port, key=key, **kwargs)
        return "session"

    monkeypatch.setattr(app_main.SshRemoteSession, "connect", fake_connect)
    factory = app_main.make_session_factory(Path("C:/data/id_ed25519"))
    instance = Instance(id=1, status="running", ssh_host="h", ssh_port=22, dph_total=0.2, label="x")
    assert factory(instance) == "session"
    assert seen["host"] == "h" and seen["port"] == 22
    assert seen["key"] == str(Path("C:/data/id_ed25519"))
    assert seen["timeout_s"] == app_main.SSH_CONNECT_TIMEOUT_S


def test_credential_vast_uses_the_current_saved_key(monkeypatch: pytest.MonkeyPatch) -> None:
    keys: list[str] = []

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            keys.append(api_key)

        def get_balance(self) -> float:
            return 4.0

    monkeypatch.setattr(app_main, "VastClient", FakeClient)
    creds = CredentialStore(backend=FakeBackend())
    creds.set(VAST_API_KEY, "first-key")
    vast = app_main.CredentialVast(creds)
    assert vast.get_balance() == 4.0
    creds.set(VAST_API_KEY, "second-key")
    assert vast.get_balance() == 4.0
    assert keys == ["first-key", "second-key"]
    assert "first-key" not in repr(vast)


def test_make_probe_only_fetches_ytdlp_for_links(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[bool] = []

    def fake_probe(source: str, is_youtube: bool, ffmpeg: str, ytdlp: Path) -> float:
        return 60.0

    monkeypatch.setattr(app_main, "probe_duration_s", fake_probe)

    def ytdlp() -> Path:
        fetched.append(True)
        return Path("yt-dlp.exe")

    probe = app_main.make_probe(lambda: "ff", ytdlp)
    assert probe(r"C:\a.mp3", False) == 60.0
    assert fetched == []
    assert probe("https://youtu.be/x", True) == 60.0
    assert fetched == [True]
