from pathlib import Path

from common.jsonio import read_json, write_json_atomic
from common.protocol import RemoteStatus, max_runtime_s


def test_status_roundtrip() -> None:
    s = RemoteStatus(
        "running",
        "labeling",
        70,
        "Diarizing chunk 7/12",
        chunk=7,
        chunks=12,
        updated_at=5.0,
    )
    assert RemoteStatus.from_json(s.to_json()) == s


def test_max_runtime_five_hours() -> None:
    assert max_runtime_s(18000.0) == 20700.0


def test_max_runtime_has_three_hour_floor() -> None:
    assert max_runtime_s(60.0) == 10800.0


def test_write_json_atomic_leaves_only_final_file(tmp_path: Path) -> None:
    write_json_atomic(tmp_path / "a.json", {"x": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
    assert read_json(tmp_path / "a.json") == {"x": 1}


def test_read_json_returns_none_for_missing_and_corrupt(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    assert read_json(tmp_path / "missing.json") is None
    assert read_json(tmp_path / "bad.json") is None
