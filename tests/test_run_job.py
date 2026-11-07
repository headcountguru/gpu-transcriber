"""Headless job runner (remote/run_job.py) driven by fake pipeline steps."""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import run_job as rj
import soundfile as sf
from aligner import align_words_with_speakers
from formatters import write_outputs

from common.jsonio import read_json
from common.protocol import (
    JOB_CHECKPOINT_DIR,
    JOB_INPUT_DIR,
    JOB_OUTPUT_DIR,
    OUTPUT_FILES,
    STATUS_FILE,
    RemoteStatus,
)

SAMPLE_RATE = 16000
TOKEN = "hf_supersecrettoken123"

WORDS = [
    {"word": " Hello", "start": 0.0, "end": 0.5},
    {"word": " there", "start": 0.5, "end": 1.0},
    {"word": " friend", "start": 1.2, "end": 1.8},
]
TURNS = [
    {"start": 0.0, "end": 1.1, "speaker": "SPEAKER_00"},
    {"start": 1.1, "end": 2.0, "speaker": "SPEAKER_01"},
]


class FakeSteps:
    """Records calls and produces canned results; real aligner/formatters are pure Python."""

    def __init__(self, seconds: float = 2.0) -> None:
        self.seconds = seconds
        self.temp_wavs: list[str] = []
        self.transcribe_calls = 0
        self.diarize_kwargs: dict = {}
        self.chunks = (1, 3)
        self.transcribe_error: Exception | None = None
        self.diarize_error: Exception | None = None

    def extract_audio(self, input_file: str) -> str:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        shutil.copy(input_file, tmp)
        self.temp_wavs.append(tmp)
        return tmp

    def transcribe_audio(
        self,
        audio_path: str,
        model_size: str = "large-v3-turbo",
        language: str | None = None,
        device: str = "cuda:0",
        progress_callback: Callable[[str, int, str], None] | None = None,
    ) -> tuple[list[dict], float, str]:
        self.transcribe_calls += 1
        if self.transcribe_error:
            raise self.transcribe_error
        if progress_callback:
            progress_callback("transcription", 30, "start")
            progress_callback("transcription", 45, "half")
            progress_callback("transcription", 60, "end")
        return [dict(w) for w in WORDS], self.seconds, "en"

    def chunk_and_diarize(
        self,
        audio_path: str,
        hf_token: str,
        device: str,
        progress_callback: Callable[[str, int, str], None] | None = None,
        checkpoint_dir: str | None = None,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        self.diarize_kwargs = {
            "audio_path": audio_path,
            "hf_token": hf_token,
            "device": device,
            "checkpoint_dir": checkpoint_dir,
        }
        if self.diarize_error:
            raise self.diarize_error
        if on_chunk:
            for i in range(1, self.chunks[1] + 1):
                on_chunk(i, self.chunks[1])
        return [dict(t) for t in TURNS]

    def apply_forced_alignment(
        self, words: list[dict], audio_path: str, language: str, device: str = "cuda"
    ) -> list[dict]:
        return words

    def as_pipeline_steps(self) -> rj.PipelineSteps:
        return rj.PipelineSteps(
            extract_audio=self.extract_audio,
            transcribe_audio=self.transcribe_audio,
            chunk_and_diarize=self.chunk_and_diarize,
            apply_forced_alignment=self.apply_forced_alignment,
            align_words_with_speakers=align_words_with_speakers,
            write_outputs=write_outputs,
        )


def make_job(tmp_path: Path, seconds: float = 2.0, name: str = "talk.wav") -> Path:
    job = tmp_path / "job"
    (job / JOB_INPUT_DIR).mkdir(parents=True)
    samples = np.zeros(int(SAMPLE_RATE * seconds), dtype="float32")
    sf.write(str(job / JOB_INPUT_DIR / name), samples, SAMPLE_RATE)
    return job


def read_status(job: Path) -> RemoteStatus:
    data = read_json(job / STATUS_FILE)
    assert isinstance(data, dict)
    return RemoteStatus(**data)


@pytest.fixture
def written_statuses(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture every status.json payload while still writing it."""
    seen: list[dict] = []
    real = rj.write_json_atomic

    def spy(path: Path, obj: object) -> None:
        if path.name == STATUS_FILE:
            seen.append(dict(obj))  # type: ignore[call-overload]
        real(path, obj)

    monkeypatch.setattr(rj, "write_json_atomic", spy)
    return seen


def test_run_job_writes_three_outputs_and_done_status(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    steps = FakeSteps()

    rj.run_job(str(job), TOKEN, device="cuda", steps=steps.as_pipeline_steps())

    for name in OUTPUT_FILES:
        assert (job / JOB_OUTPUT_DIR / name).is_file()
    status = read_status(job)
    assert status.state == "done"
    assert status.stage == "done"
    assert status.percent == 100
    assert "Hello" in (job / JOB_OUTPUT_DIR / "transcript.txt").read_text(encoding="utf-8")
    assert steps.temp_wavs and not Path(steps.temp_wavs[0]).exists()


def test_run_job_reports_stages_in_order_with_chunk_progress(
    tmp_path: Path, written_statuses: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter(range(0, 10_000, 5))  # every status write is >1 s after the previous one
    monkeypatch.setattr(rj, "_monotonic", lambda: float(next(clock)))
    job = make_job(tmp_path)

    rj.run_job(str(job), TOKEN, steps=FakeSteps().as_pipeline_steps())

    stages = [s["stage"] for s in written_statuses]
    assert list(dict.fromkeys(stages)) == [
        "preparing",
        "transcribing",
        "labeling",
        "aligning",
        "writing",
        "done",
    ]
    percents = [s["percent"] for s in written_statuses]
    assert percents == sorted(percents)
    by_stage = {s["stage"]: s["percent"] for s in reversed(written_statuses)}
    assert by_stage["preparing"] == 5
    assert by_stage["aligning"] == 90
    assert by_stage["writing"] == 95
    transcribing = [s["percent"] for s in written_statuses if s["stage"] == "transcribing"]
    assert min(transcribing) >= 10 and max(transcribing) <= 60
    labeling = [s for s in written_statuses if s["stage"] == "labeling"]
    assert labeling[0]["percent"] == 65
    assert (labeling[-1]["chunk"], labeling[-1]["chunks"]) == (3, 3)
    assert labeling[-1]["percent"] == 85
    assert all(s["state"] == "running" for s in written_statuses[:-1])
    assert all(s["updated_at"] > 0 for s in written_statuses)


def test_run_job_passes_per_job_checkpoint_dir_and_token_to_diarizer(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    steps = FakeSteps()

    rj.run_job(str(job), TOKEN, device="cuda:0", steps=steps.as_pipeline_steps())

    assert steps.diarize_kwargs["checkpoint_dir"] == str(job / JOB_CHECKPOINT_DIR)
    assert steps.diarize_kwargs["hf_token"] == TOKEN
    assert steps.diarize_kwargs["device"] == "cuda:0"
    for path in job.rglob("*"):
        if path.is_file():
            assert TOKEN not in path.read_text(encoding="utf-8", errors="ignore")


def test_run_job_writes_words_checkpoint_after_transcription(tmp_path: Path) -> None:
    job = make_job(tmp_path)

    rj.run_job(str(job), TOKEN, steps=FakeSteps(seconds=2.0).as_pipeline_steps())

    saved = read_json(job / JOB_CHECKPOINT_DIR / "words.json")
    assert saved == {"words": WORDS, "duration": 2.0, "language": "en"}


def test_run_job_skips_transcription_when_words_checkpoint_exists(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    steps = FakeSteps()
    steps.transcribe_error = AssertionError("transcribe must not be called")
    (job / JOB_CHECKPOINT_DIR).mkdir()
    rj.write_json_atomic(
        job / JOB_CHECKPOINT_DIR / "words.json",
        {"words": WORDS, "duration": 2.0, "language": "en"},
    )

    rj.run_job(str(job), TOKEN, steps=steps.as_pipeline_steps())

    assert steps.transcribe_calls == 0
    assert read_status(job).state == "done"
    assert (job / JOB_OUTPUT_DIR / "transcript.srt").is_file()


@pytest.mark.parametrize(
    "bad",
    [
        {"words": "nope", "duration": 2.0, "language": "en"},
        {"words": [], "duration": "x", "language": "en"},
        {"words": [], "duration": 2.0},
        ["not", "a", "dict"],
    ],
)
def test_run_job_retranscribes_when_words_checkpoint_is_malformed(
    tmp_path: Path, bad: object
) -> None:
    job = make_job(tmp_path)
    steps = FakeSteps()
    (job / JOB_CHECKPOINT_DIR).mkdir()
    rj.write_json_atomic(job / JOB_CHECKPOINT_DIR / "words.json", bad)

    rj.run_job(str(job), TOKEN, steps=steps.as_pipeline_steps())

    assert steps.transcribe_calls == 1
    assert read_status(job).state == "done"


def test_run_job_failure_writes_failed_status_without_token(tmp_path: Path) -> None:
    job = make_job(tmp_path)
    steps = FakeSteps()
    steps.diarize_error = RuntimeError(f"bad {TOKEN}")

    with pytest.raises(RuntimeError):
        rj.run_job(str(job), TOKEN, steps=steps.as_pipeline_steps())

    status = read_status(job)
    assert status.state == "failed"
    assert status.stage == "labeling"
    assert status.error is not None
    assert TOKEN not in status.error
    assert "***" in status.error
    assert TOKEN not in (job / STATUS_FILE).read_text(encoding="utf-8")
    assert not Path(steps.temp_wavs[0]).exists()
    assert (job / JOB_CHECKPOINT_DIR / "words.json").is_file()  # kept for resume


def test_run_job_handles_three_second_audio(tmp_path: Path) -> None:
    job = make_job(tmp_path, seconds=3.0)

    rj.run_job(str(job), TOKEN, steps=FakeSteps(seconds=3.0).as_pipeline_steps())

    for name in OUTPUT_FILES:
        assert (job / JOB_OUTPUT_DIR / name).is_file()
    assert read_status(job).state == "done"


@pytest.mark.parametrize("names", [[], ["a.wav", "b.wav"]])
def test_run_job_requires_exactly_one_input_file(tmp_path: Path, names: list[str]) -> None:
    job = make_job(tmp_path)
    (job / JOB_INPUT_DIR / "talk.wav").unlink()
    for name in names:
        (job / JOB_INPUT_DIR / name).write_bytes(b"x")

    with pytest.raises(RuntimeError, match="exactly one"):
        rj.run_job(str(job), TOKEN, steps=FakeSteps().as_pipeline_steps())

    assert read_status(job).state == "failed"


def test_status_writes_are_throttled_within_a_stage(
    tmp_path: Path, written_statuses: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rj, "_monotonic", lambda: 100.0)  # frozen clock
    job = make_job(tmp_path)
    steps = FakeSteps()
    steps.chunks = (1, 50)

    rj.run_job(str(job), TOKEN, steps=steps.as_pipeline_steps())

    # One write per stage change plus the terminal "done"; nothing for the 50 chunk ticks.
    assert len(written_statuses) == 6
    assert read_status(job).state == "done"


def test_run_job_without_steps_applies_patches_and_uses_default_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    fake = FakeSteps()
    monkeypatch.setattr(rj, "apply_compat_patches", lambda: calls.append("patches"))
    monkeypatch.setattr(
        rj, "default_steps", lambda: calls.append("steps") or fake.as_pipeline_steps()
    )
    job = make_job(tmp_path)

    rj.run_job(str(job), TOKEN)

    assert calls == ["patches", "steps"]
    assert read_status(job).state == "done"


def test_default_steps_imports_real_modules_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    names = {
        "audio_extractor": "extract_audio",
        "transcriber": "transcribe_audio",
        "chunker": "chunk_and_diarize",
        "forced_aligner": "apply_forced_alignment",
        "aligner": "align_words_with_speakers",
        "formatters": "write_outputs",
    }
    for module_name, func in names.items():
        stub = types.ModuleType(module_name)
        setattr(stub, func, lambda *a, _n=func, **k: _n)
        monkeypatch.setitem(sys.modules, module_name, stub)

    steps = rj.default_steps()

    assert steps.extract_audio() == "extract_audio"
    assert steps.transcribe_audio() == "transcribe_audio"
    assert steps.chunk_and_diarize() == "chunk_and_diarize"
    assert steps.apply_forced_alignment() == "apply_forced_alignment"
    assert steps.align_words_with_speakers() == "align_words_with_speakers"
    assert steps.write_outputs() == "write_outputs"


def test_run_job_module_does_not_import_heavy_packages() -> None:
    code = (
        "import sys; sys.path[:0] = ['.', 'remote']; import run_job; "
        "heavy = {'torch', 'faster_whisper', 'pyannote', 'torchaudio', 'lightning_fabric'}; "
        "sys.exit(1 if heavy & set(sys.modules) else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_release_gpu_memory_empties_cache_when_torch_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emptied: list[bool] = []
    cuda = types.SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: emptied.append(True)
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))

    rj.release_gpu_memory()

    assert emptied == [True]


def test_release_gpu_memory_is_a_noop_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    rj.release_gpu_memory()  # must not import torch or raise

    assert "torch" not in sys.modules


def test_main_requires_hf_token_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert rj.main(["--job-dir", str(make_job(tmp_path))]) == 2


def test_main_treats_empty_token_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "")

    assert rj.main(["--job-dir", str(make_job(tmp_path))]) == 2


def test_main_requires_job_dir_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    assert rj.main([]) == 2


def test_main_returns_zero_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(rj, "apply_compat_patches", lambda: None)
    monkeypatch.setattr(rj, "default_steps", lambda: FakeSteps().as_pipeline_steps())
    job = make_job(tmp_path)

    assert rj.main(["--job-dir", str(job)]) == 0
    assert read_status(job).state == "done"


def test_main_returns_one_on_failure_and_never_logs_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(rj, "apply_compat_patches", lambda: None)
    steps = FakeSteps()
    steps.diarize_error = RuntimeError(f"auth failed for {TOKEN}")
    monkeypatch.setattr(rj, "default_steps", lambda: steps.as_pipeline_steps())
    job = make_job(tmp_path)

    with caplog.at_level(logging.DEBUG):
        code = rj.main(["--job-dir", str(job)])

    assert code == 1
    assert read_status(job).state == "failed"
    assert TOKEN not in caplog.text
