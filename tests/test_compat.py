"""Runtime compatibility patches in remote/compat.py (all targets are stubbed)."""

import logging
import sys
import types

import compat
import numpy as np
import pytest
import soundfile as sf

SAMPLE_RATE = 16000


def _stub(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _block_all_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every patch target unimportable (a None entry raises ImportError)."""
    for name in (
        "huggingface_hub",
        "lightning_fabric.utilities.cloud_io",
        "torchaudio",
        "soundfile",
        "torch",
    ):
        monkeypatch.setitem(sys.modules, name, None)


def test_compat_adds_missing_torchaudio_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_all_targets(monkeypatch)
    torchaudio = _stub(monkeypatch, "torchaudio")
    monkeypatch.setitem(sys.modules, "soundfile", sf)

    compat.apply_compat_patches()

    assert torchaudio.list_audio_backends() == ["soundfile"]
    assert torchaudio.get_audio_backend() == "soundfile"
    assert isinstance(torchaudio.AudioMetaData, type)
    assert callable(torchaudio.info)


def test_compat_torchaudio_info_reads_wav_via_soundfile(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _block_all_targets(monkeypatch)
    torchaudio = _stub(monkeypatch, "torchaudio")
    monkeypatch.setitem(sys.modules, "soundfile", sf)
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(SAMPLE_RATE, dtype="float32"), SAMPLE_RATE)

    compat.apply_compat_patches()
    meta = torchaudio.info(str(wav))

    assert meta.sample_rate == SAMPLE_RATE
    assert meta.num_frames == SAMPLE_RATE
    assert meta.num_channels == 1
    assert isinstance(meta, torchaudio.AudioMetaData)


def test_compat_leaves_existing_torchaudio_attrs_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_all_targets(monkeypatch)

    def my_backends() -> list[str]:
        return ["sox"]

    def my_info(filepath: str, backend: object = None) -> str:
        return "mine"

    torchaudio = _stub(
        monkeypatch,
        "torchaudio",
        AudioMetaData=int,
        list_audio_backends=my_backends,
        get_audio_backend=lambda: "sox",
        info=my_info,
    )

    compat.apply_compat_patches()

    assert torchaudio.AudioMetaData is int
    assert torchaudio.list_audio_backends is my_backends
    assert torchaudio.info is my_info


def test_compat_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_all_targets(monkeypatch)
    hub = _stub(monkeypatch, "huggingface_hub", login=lambda token=None, **kw: "real")
    cloud_io = _stub(monkeypatch, "lightning_fabric.utilities.cloud_io", _load=lambda *a, **k: None)
    torchaudio = _stub(monkeypatch, "torchaudio")
    _stub(monkeypatch, "torch", load=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "soundfile", sf)

    compat.apply_compat_patches()
    first = (
        hub.login,
        cloud_io._load,
        torchaudio.AudioMetaData,
        torchaudio.list_audio_backends,
        torchaudio.get_audio_backend,
        torchaudio.info,
    )
    compat.apply_compat_patches()
    second = (
        hub.login,
        cloud_io._load,
        torchaudio.AudioMetaData,
        torchaudio.list_audio_backends,
        torchaudio.get_audio_backend,
        torchaudio.info,
    )

    assert all(a is b for a, b in zip(first, second, strict=True))


def test_compat_makes_huggingface_login_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_all_targets(monkeypatch)
    calls: list[object] = []
    hub = _stub(monkeypatch, "huggingface_hub", login=lambda token=None, **kw: calls.append(token))

    compat.apply_compat_patches()

    assert hub.login("secret", add_to_git_credential=True) is None
    assert calls == []


def test_compat_forces_weights_only_false_in_lightning_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_all_targets(monkeypatch)
    loads: list[dict] = []
    _stub(monkeypatch, "torch", load=lambda path, **kw: loads.append({"path": path, **kw}) or "ok")
    cloud_io = _stub(monkeypatch, "lightning_fabric.utilities.cloud_io", _load=None)

    compat.apply_compat_patches()
    result = cloud_io._load("model.ckpt", map_location="cpu", weights_only=True)

    assert result == "ok"
    assert loads == [{"path": "model.ckpt", "map_location": "cpu", "weights_only": False}]


def test_compat_skips_unimportable_targets_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _block_all_targets(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="compat"):
        compat.apply_compat_patches()

    text = caplog.text
    for name in ("huggingface_hub", "lightning_fabric", "torchaudio"):
        assert name in text


def test_compat_warns_when_soundfile_missing_for_torchaudio_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _block_all_targets(monkeypatch)
    torchaudio = _stub(monkeypatch, "torchaudio")

    with caplog.at_level(logging.WARNING, logger="compat"):
        compat.apply_compat_patches()

    assert not hasattr(torchaudio, "info")
    assert torchaudio.list_audio_backends() == ["soundfile"]
    assert "soundfile" in caplog.text
