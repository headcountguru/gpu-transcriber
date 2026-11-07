"""Checkpointing and resume behaviour of remote/chunker.py."""

import importlib
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from common.jsonio import read_json, write_json_atomic

SAMPLE_RATE = 16000


@pytest.fixture
def chunker(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import remote/chunker.py with a stub ``diarizer`` (pyannote is not installed here)."""
    stub = types.ModuleType("diarizer")
    stub.diarize_audio = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diarizer", stub)
    monkeypatch.delitem(sys.modules, "chunker", raising=False)
    module = importlib.import_module("chunker")
    monkeypatch.setattr(module, "CHUNK_THRESHOLD_MINUTES", 0.05)
    monkeypatch.setattr(module, "CHUNK_MINUTES", 0.1)
    monkeypatch.setattr(module, "OVERLAP_MINUTES", 0.02)
    yield module
    sys.modules.pop("chunker", None)


class FakeDiarizer:
    """Counts calls and returns one SPEAKER_00 turn spanning the whole chunk."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, path: str, hf_token: str, device: str) -> list[dict]:
        self.calls += 1
        with sf.SoundFile(path) as f:
            length = len(f) / f.samplerate
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": length}]


def write_silent_wav(path: Path, seconds: float) -> str:
    sf.write(str(path), np.zeros(int(seconds * SAMPLE_RATE), dtype="float32"), SAMPLE_RATE)
    return str(path)


@pytest.fixture
def fake(chunker: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> FakeDiarizer:
    diarizer = FakeDiarizer()
    monkeypatch.setattr(chunker, "diarize_audio", diarizer)
    return diarizer


@pytest.fixture
def long_audio(tmp_path: Path) -> str:
    return write_silent_wav(tmp_path / "long.wav", 10.0)


def window_count(chunker: types.ModuleType) -> int:
    return len(chunker.plan_windows(10.0, 0.1 * 60, 0.02 * 60))


def test_plan_windows_matches_existing_behaviour(chunker: types.ModuleType) -> None:
    assert chunker.plan_windows(100, 30, 2) == [(0, 30), (28, 58), (56, 86), (84, 100)]


def test_checkpoints_written_one_per_chunk(
    chunker: types.ModuleType, fake: FakeDiarizer, long_audio: str, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    n = window_count(chunker)
    assert n > 1

    chunker.chunk_and_diarize(long_audio, "tok", "cpu", checkpoint_dir=str(ckpt))

    assert sorted(p.name for p in ckpt.glob("diar_*.json")) == [
        f"diar_{i:03d}.json" for i in range(n)
    ]
    first = read_json(ckpt / "diar_000.json")
    assert first["index"] == 0
    assert first["window"][0] == 0
    assert first["turns"][0]["speaker"] == "SPEAKER_00"
    assert fake.calls == n


def test_resume_skips_finished_chunks(
    chunker: types.ModuleType,
    fake: FakeDiarizer,
    long_audio: str,
    tmp_path: Path,
) -> None:
    fresh_dir = tmp_path / "fresh"
    fresh_dir.mkdir()
    fresh = chunker.chunk_and_diarize(long_audio, "tok", "cpu", checkpoint_dir=str(fresh_dir))
    n = fake.calls

    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    write_json_atomic(resume_dir / "diar_000.json", read_json(fresh_dir / "diar_000.json"))
    fake.calls = 0

    resumed = chunker.chunk_and_diarize(long_audio, "tok", "cpu", checkpoint_dir=str(resume_dir))

    assert fake.calls == n - 1
    assert resumed == fresh


def test_short_audio_is_one_checkpoint(
    chunker: types.ModuleType, fake: FakeDiarizer, tmp_path: Path
) -> None:
    short = write_silent_wav(tmp_path / "short.wav", 2.0)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()

    result = chunker.chunk_and_diarize(short, "tok", "cpu", checkpoint_dir=str(ckpt))

    assert [p.name for p in ckpt.glob("diar_*.json")] == ["diar_000.json"]
    assert fake.calls == 1
    saved = read_json(ckpt / "diar_000.json")
    assert saved["window"] == [0, pytest.approx(2.0)]
    assert result == saved["turns"]


def test_short_audio_resume_skips_diarization(
    chunker: types.ModuleType, fake: FakeDiarizer, tmp_path: Path
) -> None:
    short = write_silent_wav(tmp_path / "short.wav", 2.0)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    first = chunker.chunk_and_diarize(short, "tok", "cpu", checkpoint_dir=str(ckpt))
    fake.calls = 0

    again = chunker.chunk_and_diarize(short, "tok", "cpu", checkpoint_dir=str(ckpt))

    assert fake.calls == 0
    assert again == first


def test_on_chunk_counts_skipped_chunks(
    chunker: types.ModuleType, fake: FakeDiarizer, long_audio: str, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    n = window_count(chunker)
    chunker.chunk_and_diarize(long_audio, "tok", "cpu", checkpoint_dir=str(ckpt))
    (ckpt / f"diar_{n - 1:03d}.json").unlink()  # leave 0..n-2 on disk, redo the last one

    calls: list[tuple[int, int]] = []
    chunker.chunk_and_diarize(
        long_audio,
        "tok",
        "cpu",
        checkpoint_dir=str(ckpt),
        on_chunk=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(i, n) for i in range(1, n + 1)]


def test_without_checkpoint_dir_behaviour_is_unchanged(
    chunker: types.ModuleType, fake: FakeDiarizer, long_audio: str, tmp_path: Path
) -> None:
    result = chunker.chunk_and_diarize(long_audio, "tok", "cpu")

    assert fake.calls == window_count(chunker)
    assert result
    assert not list(tmp_path.glob("diar_*.json"))


def test_unreadable_checkpoint_is_redone(
    chunker: types.ModuleType, fake: FakeDiarizer, long_audio: str, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "diar_000.json").write_text("{not json", encoding="utf-8")

    chunker.chunk_and_diarize(long_audio, "tok", "cpu", checkpoint_dir=str(ckpt))

    assert fake.calls == window_count(chunker)
    assert read_json(ckpt / "diar_000.json")["index"] == 0


GOOD_TURN = {"speaker": "SPEAKER_09", "start": 0.0, "end": 1.0}


def corrupt_checkpoint(chunker: types.ModuleType, ckpt: Path, **overrides: object) -> dict:
    """Write diar_000.json for chunk 0 of the long audio, with some fields overridden."""
    window = chunker.plan_windows(10.0, 0.1 * 60, 0.02 * 60)[0]
    payload = {"index": 0, "window": list(window), "turns": [GOOD_TURN]} | overrides
    write_json_atomic(ckpt / "diar_000.json", payload)
    return payload


def assert_chunk_zero_redone(
    chunker: types.ModuleType, fake: FakeDiarizer, long_audio: str, ckpt: Path, tmp_path: Path
) -> None:
    fresh_dir = tmp_path / "fresh"
    fresh_dir.mkdir()
    fresh = chunker.chunk_and_diarize(long_audio, "tok", "cpu", checkpoint_dir=str(fresh_dir))
    fake.calls = 0

    result = chunker.chunk_and_diarize(long_audio, "tok", "cpu", checkpoint_dir=str(ckpt))

    assert fake.calls == window_count(chunker)  # chunk 0 was diarized again, not loaded
    assert result == fresh
    assert read_json(ckpt / "diar_000.json") == read_json(fresh_dir / "diar_000.json")


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"window": [0, 999.0]}, id="wrong-window-end"),
        pytest.param({"window": [5.0, 6.0]}, id="wrong-window-start"),
        pytest.param({"window": None}, id="window-missing"),
        pytest.param({"index": 3}, id="wrong-index"),
        pytest.param({"index": None}, id="index-missing"),
        pytest.param({"turns": {"speaker": "SPEAKER_09"}}, id="turns-not-a-list"),
        pytest.param({"turns": None}, id="turns-missing"),
        pytest.param({"turns": [{"speaker": "SPEAKER_09", "start": 0.0}]}, id="turn-lacks-end"),
        pytest.param({"turns": [{"start": 0.0, "end": 1.0}]}, id="turn-lacks-speaker"),
        pytest.param({"turns": [{"speaker": "SPEAKER_09", "end": 1.0}]}, id="turn-lacks-start"),
        pytest.param({"turns": [GOOD_TURN, "not-a-turn"]}, id="turn-not-a-dict"),
    ],
)
def test_mismatched_checkpoint_is_redone(
    chunker: types.ModuleType,
    fake: FakeDiarizer,
    long_audio: str,
    tmp_path: Path,
    overrides: dict,
) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    corrupt_checkpoint(chunker, ckpt, **overrides)

    assert_chunk_zero_redone(chunker, fake, long_audio, ckpt, tmp_path)


def test_checkpoint_of_a_non_object_json_is_redone(
    chunker: types.ModuleType, fake: FakeDiarizer, long_audio: str, tmp_path: Path
) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    write_json_atomic(ckpt / "diar_000.json", [GOOD_TURN])

    assert_chunk_zero_redone(chunker, fake, long_audio, ckpt, tmp_path)


def test_long_audio_checkpoint_is_not_reused_for_short_audio(
    chunker: types.ModuleType, fake: FakeDiarizer, tmp_path: Path
) -> None:
    short = write_silent_wav(tmp_path / "short.wav", 2.0)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    corrupt_checkpoint(chunker, ckpt)  # window is chunk 0 of a 10 s file, not [0, 2]

    result = chunker.chunk_and_diarize(short, "tok", "cpu", checkpoint_dir=str(ckpt))

    assert fake.calls == 1
    assert result == [{"speaker": "SPEAKER_00", "start": 0.0, "end": pytest.approx(2.0)}]
    saved = read_json(ckpt / "diar_000.json")
    assert saved["window"] == [0, pytest.approx(2.0)]
    assert saved["turns"] == result
