"""Tests for the job model and on-disk job store."""

import json
import logging
from pathlib import Path

import pytest

from app.jobs import (
    TERMINAL_STATES,
    Job,
    JobState,
    JobStore,
    state_for_stage,
)


def make_job(job_id: str = "j1", created_at: float = 100.0, **changes: object) -> Job:
    job = Job(
        id=job_id,
        name="Interview",
        source="C:/audio/interview.mp3",
        is_youtube=False,
        state=JobState.PREPARING,
        created_at=created_at,
    )
    return job.with_(**changes) if changes else job


def test_job_store_roundtrip_and_newest_first(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    older = make_job("old", created_at=100.0, percent=10, chunk=1, chunks=4)
    newer = make_job("new", created_at=200.0, state=JobState.DONE, cost_so_far=1.25)
    store.save(older)
    store.save(newer)

    assert store.get("old") == older
    assert store.get("new") == newer
    assert [j.id for j in store.load_all()] == ["new", "old"]
    stored = json.loads((tmp_path / "new" / "job.json").read_text(encoding="utf-8"))
    assert stored["state"] == "DONE"


def test_save_creates_job_folders_and_overwrites(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.save(make_job("j1"))
    store.save(make_job("j1", percent=50))

    job_dir = store.job_dir("j1")
    assert job_dir == tmp_path / "j1"
    for sub in ("audio", "checkpoints", "output"):
        assert (job_dir / sub).is_dir()
    assert store.get("j1").percent == 50
    assert not list(job_dir.glob("*.tmp"))


def test_store_skips_corrupt_job_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store = JobStore(tmp_path)
    store.save(make_job("good", created_at=1.0))
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "job.json").write_text("{not json", encoding="utf-8")
    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    (wrong_dir / "job.json").write_text(json.dumps({"id": "wrong"}), encoding="utf-8")
    (tmp_path / "stray.txt").write_text("x", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        jobs = store.load_all()

    assert [j.id for j in jobs] == ["good"]
    skipped = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Skipping")]
    assert any(" bad:" in m for m in skipped) and any(" wrong:" in m for m in skipped)


def test_load_all_on_missing_root_returns_empty(tmp_path: Path) -> None:
    assert JobStore(tmp_path / "nope").load_all() == []


def test_load_tolerates_missing_optional_and_unknown_keys(tmp_path: Path) -> None:
    folder = tmp_path / "j1"
    folder.mkdir()
    data = {
        "id": "j1",
        "name": "n",
        "source": "s",
        "is_youtube": True,
        "state": "TRANSCRIBING",
        "created_at": 5.0,
        "added_in_a_future_version": [1, 2],
    }
    (folder / "job.json").write_text(json.dumps(data), encoding="utf-8")

    job = JobStore(tmp_path).get("j1")

    assert job.state is JobState.TRANSCRIBING
    assert job.percent == 0
    assert job.chunk is None
    assert job.retries == 0


def test_get_missing_job_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        JobStore(tmp_path).get("absent")


@pytest.mark.parametrize("bad_id", ["", "..", "a/b", "a\\b", "../x", ".", "x/..", "C:evil"])
def test_store_rejects_unsafe_job_ids(tmp_path: Path, bad_id: str) -> None:
    store = JobStore(tmp_path)
    with pytest.raises(ValueError):
        store.job_dir(bad_id)
    with pytest.raises(ValueError):
        store.get(bad_id)
    with pytest.raises(ValueError):
        store.save(make_job(bad_id))


def test_with_returns_new_job_and_rejects_unknown_field() -> None:
    job = make_job()
    changed = job.with_(percent=7, state=JobState.UPLOADING)

    assert changed.percent == 7 and changed.state is JobState.UPLOADING
    assert job.percent == 0 and job.state is JobState.PREPARING
    with pytest.raises(TypeError):
        job.with_(nonsense=1)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("preparing", JobState.STARTING_MACHINE),
        ("transcribing", JobState.TRANSCRIBING),
        ("labeling", JobState.LABELING),
        ("aligning", JobState.LABELING),
        ("writing", JobState.SAVING),
        ("done", JobState.SAVING),
    ],
)
def test_state_for_stage_mapping(stage: str, expected: JobState) -> None:
    assert state_for_stage(stage) is expected


def test_state_for_stage_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown remote stage"):
        state_for_stage("rewinding")


def test_terminal_states_exclude_running_states() -> None:
    assert TERMINAL_STATES == frozenset(
        {JobState.DONE, JobState.FAILED, JobState.STOPPED_AT_CAP, JobState.CANCELLED}
    )
    for running in (
        JobState.PREPARING,
        JobState.UPLOADING,
        JobState.STARTING_MACHINE,
        JobState.TRANSCRIBING,
        JobState.LABELING,
        JobState.SAVING,
    ):
        assert running not in TERMINAL_STATES


def _write_raw(root: Path, job_id: str, raw: str) -> None:
    folder = root / job_id
    folder.mkdir()
    (folder / "job.json").write_text(raw, encoding="utf-8")


def _job_json(**overrides: object) -> str:
    data: dict[str, object] = {
        "id": "x",
        "name": "n",
        "source": "s",
        "is_youtube": False,
        "state": "PREPARING",
        "created_at": 5.0,
    }
    data.update(overrides)
    return json.dumps(data)


_BAD_FILES = {
    "created_at_string": _job_json(created_at="x"),
    "created_at_null": _job_json(created_at=None),
    "created_at_bool": _job_json(created_at=True),
    "created_at_nan": _job_json(created_at=float("nan")),
    "created_at_inf": _job_json(created_at=float("inf")),
    "non_dict": "[]",
    "unknown_state": _job_json(state="EXPLODED"),
    "state_not_string": _job_json(state=3),
    "name_not_string": _job_json(name=7),
    "is_youtube_not_bool": _job_json(is_youtube="yes"),
    "percent_bool": _job_json(percent=True),
    "percent_string": _job_json(percent="50"),
    "chunk_string": _job_json(chunk="1"),
    "price_string": _job_json(price_per_hour="0.5"),
    "error_number": _job_json(error=5),
}


def test_load_all_skips_wrong_typed_files_and_keeps_good_ones(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = JobStore(tmp_path)
    store.save(make_job("good", created_at=1.0))
    for name, raw in _BAD_FILES.items():
        _write_raw(tmp_path, f"bad_{name}", raw)

    with caplog.at_level(logging.WARNING):
        jobs = store.load_all()

    assert [j.id for j in jobs] == ["good"]
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "bad_created_at_string" in messages


@pytest.mark.parametrize("name", sorted(_BAD_FILES))
def test_get_raises_value_error_for_corrupt_file(tmp_path: Path, name: str) -> None:
    _write_raw(tmp_path, "bad", _BAD_FILES[name])

    with pytest.raises(ValueError):
        JobStore(tmp_path).get("bad")


def test_get_raises_value_error_for_unparseable_json(tmp_path: Path) -> None:
    _write_raw(tmp_path, "bad", "{not json")

    with pytest.raises(ValueError):
        JobStore(tmp_path).get("bad")


def test_integer_created_at_and_pay_fields_are_accepted(tmp_path: Path) -> None:
    _write_raw(tmp_path, "ok", _job_json(created_at=5, price_per_hour=1, duration_s=60))

    job = JobStore(tmp_path).get("ok")

    assert job.created_at == 5 and job.price_per_hour == 1
