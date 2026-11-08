"""Headless job runner: transcribe, diarize, align and write outputs for one job folder.

Runs on the rented GPU machine. Reads the single uploaded file from ``<job>/input``,
keeps resumable checkpoints in ``<job>/checkpoints``, writes the three transcript files
to ``<job>/output`` and a ``status.json`` the desktop app polls over SSH.

Heavy packages (torch, faster-whisper, pyannote) are imported lazily by
:func:`default_steps` so this module imports anywhere. The Hugging Face token is only
ever held in memory and is scrubbed from any error text written to ``status.json``.
"""

import argparse
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from compat import apply_compat_patches

from common.jsonio import read_json, write_json_atomic
from common.protocol import (
    JOB_CHECKPOINT_DIR,
    JOB_INPUT_DIR,
    JOB_OUTPUT_DIR,
    STATUS_FILE,
    RemoteStatus,
    shorten_remote_text,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

WORDS_CHECKPOINT = "words.json"
STATUS_MIN_INTERVAL_S = 1.0
REDACTED = "***"
# While running, status.json is rewritten at least this often, even during one long step that
# reports no progress (a 30-minute diarization chunk, forced alignment). The app treats a status
# that has not changed for many heartbeats as a dead runner. Read at call time so tests can
# shorten it.
HEARTBEAT_S = 60.0
HEARTBEAT_THREAD_NAME = "status-heartbeat"
RUNNER_LOCK_FILE = ".runner.lock"


class RunnerAlreadyRunningError(RuntimeError):
    """Another runner holds this job folder's lock; this one did nothing."""


PERCENT_PREPARING = 5
PERCENT_TRANSCRIBE_START = 10
PERCENT_TRANSCRIBE_END = 60
PERCENT_LABEL_START = 65
PERCENT_LABEL_END = 85
PERCENT_ALIGNING = 90
PERCENT_WRITING = 95
PERCENT_DONE = 100

# The transcriber reports overall progress on the old 30..60 scale of the Kaggle pipeline.
_TRANSCRIBER_SCALE_START = 30
_TRANSCRIBER_SCALE_END = 60

# Module-level reference so tests can control the clock.
_monotonic = time.monotonic

ProgressCallback = Callable[[str, int, str], None]


@dataclass(frozen=True)
class PipelineSteps:
    """The pipeline stages as callables, so tests can substitute fakes."""

    extract_audio: Callable[[str], str]
    transcribe_audio: Callable[..., tuple[list[dict], float, str]]
    chunk_and_diarize: Callable[..., list[dict]]
    apply_forced_alignment: Callable[..., list[dict]]
    align_words_with_speakers: Callable[[list[dict], list[dict]], list[dict]]
    write_outputs: Callable[[list[dict], str], object]


def default_steps() -> PipelineSteps:
    """Import the real pipeline modules (heavy: torch, faster-whisper, pyannote)."""
    from aligner import align_words_with_speakers
    from audio_extractor import extract_audio
    from chunker import chunk_and_diarize
    from forced_aligner import apply_forced_alignment
    from formatters import write_outputs
    from transcriber import transcribe_audio

    return PipelineSteps(
        extract_audio=extract_audio,
        transcribe_audio=transcribe_audio,
        chunk_and_diarize=chunk_and_diarize,
        apply_forced_alignment=apply_forced_alignment,
        align_words_with_speakers=align_words_with_speakers,
        write_outputs=write_outputs,
    )


def release_gpu_memory() -> None:
    """Free cached GPU memory between the transcription and diarization models."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def redact(text: str, token: str) -> str:
    """Replace every occurrence of the secret ``token`` in ``text`` with ``***``."""
    return text.replace(token, REDACTED) if token else text


class _StatusReporter:
    """Writes ``status.json``, at most once a second unless the stage changes.

    ``beat`` rewrites the last running status with a fresh ``updated_at`` (see ``HEARTBEAT_S``).
    Writes are serialised by a lock because the heartbeat runs on its own thread.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._stage = ""
        self._percent = 0
        self._last_write = float("-inf")
        self._last: RemoteStatus | None = None
        self._lock = threading.Lock()

    def beat(self) -> None:
        with self._lock:
            last = self._last
            if last is None or last.state != "running":
                return
            self._write_locked(last.state, last.message, last.chunk, last.chunks, None, None)

    def update(
        self,
        stage: str,
        percent: int,
        message: str,
        chunk: int | None = None,
        chunks: int | None = None,
    ) -> None:
        stage_changed = stage != self._stage
        self._stage = stage
        self._percent = percent
        now = _monotonic()
        if not stage_changed and now - self._last_write < STATUS_MIN_INTERVAL_S:
            return
        self._write("running", message, chunk, chunks, None, now)

    def finish(self, message: str) -> None:
        self._stage = "done"
        self._percent = PERCENT_DONE
        self._write("done", message, None, None, None, _monotonic())

    def fail(self, error: str) -> None:
        self._write("failed", "Job failed", None, None, error, _monotonic())

    def _write(
        self,
        state: str,
        message: str,
        chunk: int | None,
        chunks: int | None,
        error: str | None,
        now: float,
    ) -> None:
        with self._lock:
            self._write_locked(state, message, chunk, chunks, error, now)

    def _write_locked(
        self,
        state: str,
        message: str,
        chunk: int | None,
        chunks: int | None,
        error: str | None,
        now: float | None,
    ) -> None:
        status = RemoteStatus(
            state=state,
            stage=self._stage,
            percent=self._percent,
            message=message,
            chunk=chunk,
            chunks=chunks,
            error=error,
            updated_at=time.time(),
            heartbeat_s=HEARTBEAT_S,
        )
        write_json_atomic(self._path, asdict(status))
        self._last = status
        if now is not None:  # a heartbeat does not count against the progress throttle
            self._last_write = now


class _Heartbeat:
    """Calls ``reporter.beat()`` every ``HEARTBEAT_S`` seconds until stopped."""

    def __init__(self, reporter: _StatusReporter) -> None:
        self._reporter = reporter
        self._interval = HEARTBEAT_S
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=HEARTBEAT_THREAD_NAME, daemon=True)

    def __enter__(self) -> "_Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._reporter.beat()
            except OSError as exc:
                logger.warning("Could not refresh status.json: %s", type(exc).__name__)


def _find_input(input_dir: Path) -> Path:
    files = sorted(p for p in input_dir.iterdir() if p.is_file()) if input_dir.is_dir() else []
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one input file in {input_dir}, found {len(files)}")
    return files[0]


def _load_words_checkpoint(path: Path) -> tuple[list[dict], float, str] | None:
    data = read_json(path)
    if not isinstance(data, dict):
        return None
    words, duration, language = data.get("words"), data.get("duration"), data.get("language")
    if (
        not isinstance(words, list)
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not isinstance(language, str)
    ):
        return None
    return words, float(duration), language


def _scale_transcriber_percent(percent: int) -> int:
    span_in = _TRANSCRIBER_SCALE_END - _TRANSCRIBER_SCALE_START
    span_out = PERCENT_TRANSCRIBE_END - PERCENT_TRANSCRIBE_START
    scaled = PERCENT_TRANSCRIBE_START + (percent - _TRANSCRIBER_SCALE_START) * span_out // span_in
    return max(PERCENT_TRANSCRIBE_START, min(PERCENT_TRANSCRIBE_END, scaled))


def _transcribe_or_resume(
    steps: PipelineSteps, wav_path: str, device: str, checkpoint: Path, reporter: _StatusReporter
) -> tuple[list[dict], float, str]:
    saved = _load_words_checkpoint(checkpoint)
    if saved is not None:
        reporter.update("transcribing", PERCENT_TRANSCRIBE_END, "Using saved transcription")
        return saved

    reporter.update("transcribing", PERCENT_TRANSCRIBE_START, "Transcribing audio")

    def on_progress(_stage: str, percent: int, message: str) -> None:
        reporter.update("transcribing", _scale_transcriber_percent(percent), message)

    words, duration, language = steps.transcribe_audio(
        wav_path, device=device, progress_callback=on_progress
    )
    write_json_atomic(checkpoint, {"words": words, "duration": duration, "language": language})
    return words, duration, language


def _label_speakers(
    steps: PipelineSteps,
    wav_path: str,
    hf_token: str,
    device: str,
    checkpoint_dir: Path,
    reporter: _StatusReporter,
) -> list[dict]:
    reporter.update("labeling", PERCENT_LABEL_START, "Identifying speakers")

    def on_chunk(done: int, total: int) -> None:
        span = PERCENT_LABEL_END - PERCENT_LABEL_START
        percent = PERCENT_LABEL_START + span * done // max(total, 1)
        reporter.update("labeling", percent, f"Identifying speakers ({done}/{total})", done, total)

    return steps.chunk_and_diarize(
        audio_path=wav_path,
        hf_token=hf_token,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        on_chunk=on_chunk,
    )


def _run_pipeline(
    job: Path, hf_token: str, device: str, steps: PipelineSteps, reporter: _StatusReporter
) -> None:
    checkpoint_dir = job / JOB_CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    reporter.update("preparing", PERCENT_PREPARING, "Preparing audio")
    wav_path = steps.extract_audio(str(_find_input(job / JOB_INPUT_DIR)))
    try:
        words, _duration, language = _transcribe_or_resume(
            steps, wav_path, device, checkpoint_dir / WORDS_CHECKPOINT, reporter
        )
        release_gpu_memory()

        turns = _label_speakers(steps, wav_path, hf_token, device, checkpoint_dir, reporter)
        release_gpu_memory()

        reporter.update("aligning", PERCENT_ALIGNING, "Aligning words with speakers")
        words = steps.apply_forced_alignment(words, wav_path, language or "en", device)
        segments = steps.align_words_with_speakers(words, turns)

        reporter.update("writing", PERCENT_WRITING, "Writing transcripts")
        steps.write_outputs(segments, str(job / JOB_OUTPUT_DIR))
    finally:
        Path(wav_path).unlink(missing_ok=True)
    reporter.finish("Done")


def run_job(
    job_dir: str, hf_token: str, device: str = "cuda", steps: PipelineSteps | None = None
) -> None:
    """Run the whole pipeline for ``job_dir``.

    A first status is written before the heavy imports, and a heartbeat keeps it fresh while
    the job runs. On anything that stops the job, including ``KeyboardInterrupt`` and
    ``SystemExit``, ``status.json`` is set to ``failed`` (token scrubbed, then shortened) and
    the exception is re-raised so the caller can map it to an exit code.
    """
    job = Path(job_dir)
    release = _hold_runner_lock(job)
    if release is None:  # raised before any status write: the first runner owns status.json
        raise RunnerAlreadyRunningError("Another runner is already working on this job")
    reporter = _StatusReporter(job / STATUS_FILE)
    try:
        with _Heartbeat(reporter):
            reporter.update("preparing", PERCENT_PREPARING, "Starting")
            if steps is None:
                apply_compat_patches()
                steps = default_steps()
            _run_pipeline(job, hf_token, device, steps, reporter)
    except BaseException as exc:
        error = shorten_remote_text(redact(f"{type(exc).__name__}: {exc}", hf_token))
        try:
            reporter.fail(error)
        except OSError as write_error:
            logger.error("Could not write the failed status: %s", type(write_error).__name__)
        raise
    finally:
        release()


def _hold_runner_lock(job: Path) -> Callable[[], None] | None:
    """Lock the job folder for this process; returns the release, or None if already locked.

    If the app re-attaches after a crash between launching the runner and its first status
    write, it launches again; the second runner then finds the lock and exits without touching
    anything. The lock is an ``flock`` on ``<job>/.runner.lock``, dropped by the kernel when the
    holder exits or dies. Where ``fcntl`` does not exist (Windows, where the tests run) there is
    nothing to lock against.
    """
    try:
        import fcntl
    except ImportError:
        return lambda: None
    job.mkdir(parents=True, exist_ok=True)
    handle = open(job / RUNNER_LOCK_FILE, "a", encoding="utf-8")  # noqa: SIM115 (held for the run)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle.close


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Exit codes: 0 done, 1 failed, 2 usage (bad args or no ``HF_TOKEN``)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run one transcription job headlessly.")
    parser.add_argument("--job-dir", required=True, help="Job folder (input/, checkpoints/, ...)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.error("HF_TOKEN environment variable is not set")
        return EXIT_USAGE

    try:
        run_job(args.job_dir, hf_token)
    except (Exception, KeyboardInterrupt) as exc:
        logger.error("Job failed: %s", redact(f"{type(exc).__name__}: {exc}", hf_token))
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
