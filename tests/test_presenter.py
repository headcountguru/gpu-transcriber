"""Tests for the pure presentation helpers behind the window (no Qt)."""

from pathlib import Path

import pytest

from app.jobs import Job, JobState
from app.presenter import (
    StageRow,
    alarm_text,
    check_source,
    cost_text,
    stage_rows,
    status_kind,
    suggested_cap,
    summary_line,
)


def make_job(**changes: object) -> Job:
    base = Job(
        id="job1",
        name="Interview",
        source="https://youtu.be/abc",
        is_youtube=True,
        state=JobState.PREPARING,
        created_at=1.0,
    )
    return base.with_(**changes)


def test_stage_rows_for_labeling_chunk_7_of_12() -> None:
    rows = stage_rows(make_job(state=JobState.LABELING, chunk=7, chunks=12, audio_path="a"))
    assert rows == [
        StageRow("Preparing", "done"),
        StageRow("Starting machine", "done"),
        StageRow("Uploading", "done"),
        StageRow("Transcribing", "done"),
        StageRow("Labeling speakers (chunk 7 of 12)", "active"),
        StageRow("Saving results", "pending"),
    ]


def test_stage_rows_labeling_without_chunk_numbers() -> None:
    rows = stage_rows(make_job(state=JobState.LABELING))
    assert rows[4] == StageRow("Labeling speakers", "active")


def test_stage_rows_mark_failed_stage() -> None:
    job = make_job(state=JobState.FAILED, audio_path="a", chunk=3, chunks=12, percent=40)
    rows = stage_rows(job)
    statuses = [row.status for row in rows]
    assert statuses == ["done", "done", "done", "done", "failed", "pending"]
    assert rows[4].label == "Labeling speakers (chunk 3 of 12)"


def test_stage_rows_failed_while_preparing() -> None:
    rows = stage_rows(make_job(state=JobState.FAILED))
    assert [row.status for row in rows] == ["failed"] + ["pending"] * 5


def test_stage_rows_failed_while_finding_a_machine() -> None:
    rows = stage_rows(make_job(state=JobState.FAILED, audio_path="a", duration_s=60.0))
    assert [row.status for row in rows][:3] == ["done", "failed", "pending"]


def test_stage_rows_done_marks_everything_done() -> None:
    rows = stage_rows(make_job(state=JobState.DONE, output_dir="out"))
    assert all(row.status == "done" for row in rows)


def test_stage_rows_stopped_at_cap_has_no_failed_marker() -> None:
    rows = stage_rows(make_job(state=JobState.STOPPED_AT_CAP, audio_path="a", chunk=5, chunks=9))
    assert "failed" not in [row.status for row in rows]
    assert rows[4] == StageRow("Labeling speakers (chunk 5 of 9)", "pending")
    assert rows[3].status == "done"


def test_cost_text_formats_cap() -> None:
    assert cost_text(make_job(cost_so_far=0.21, cost_cap=2.0)) == "$0.21 of $2.00 cap"


def test_cost_text_without_cap() -> None:
    assert cost_text(make_job()) == "$0.00 so far"


def test_summary_line_running_shows_stage_and_percent() -> None:
    job = make_job(state=JobState.LABELING, chunk=7, chunks=12, percent=58)
    assert summary_line(job) == "Labeling speakers (chunk 7 of 12) · 58%"


@pytest.mark.parametrize(
    ("state", "text"),
    [
        (JobState.DONE, "Done"),
        (JobState.FAILED, "Failed"),
        (JobState.STOPPED_AT_CAP, "Stopped at the spending cap"),
        (JobState.CANCELLED, "Cancelled"),
        (JobState.PREPARING, "Preparing"),
    ],
)
def test_summary_line_final_states(state: JobState, text: str) -> None:
    assert summary_line(make_job(state=state)) == text


@pytest.mark.parametrize(
    ("state", "kind"),
    [
        (JobState.LABELING, "running"),
        (JobState.DONE, "done"),
        (JobState.FAILED, "failed"),
        (JobState.STOPPED_AT_CAP, "cap"),
        (JobState.CANCELLED, "muted"),
    ],
)
def test_status_kind_is_semantic(state: JobState, kind: str) -> None:
    assert status_kind(make_job(state=state)) == kind


def test_alarm_text_only_for_final_job_with_machine() -> None:
    assert alarm_text(make_job(state=JobState.LABELING, instance_id=5)) is None
    assert alarm_text(make_job(state=JobState.FAILED)) is None
    text = alarm_text(make_job(state=JobState.FAILED, instance_id=4242))
    assert text is not None
    assert "4242" in text
    assert "may still be running and billing" in text
    assert "cloud.vast.ai" in text


def test_suggested_cap_is_above_old_cap_and_spend() -> None:
    minimum, default = suggested_cap(make_job(cost_cap=2.0, cost_so_far=2.03, banked_cost=2.03))
    assert minimum > 2.03
    assert default >= minimum
    assert default == pytest.approx(4.0)


def test_check_source_empty_is_not_ok_without_error() -> None:
    result = check_source("   ")
    assert not result.ok
    assert result.error is None


def test_check_source_youtube_link() -> None:
    result = check_source(" https://www.youtube.com/watch?v=abc123 ")
    assert result.ok
    assert result.is_youtube
    assert result.source == "https://www.youtube.com/watch?v=abc123"


def test_check_source_playlist_link_gives_plain_error() -> None:
    result = check_source("https://www.youtube.com/playlist?list=PL123")
    assert not result.ok
    assert result.error is not None
    assert "single video" in result.error


def test_check_source_path_needs_a_file_check_and_strips_quotes(tmp_path: Path) -> None:
    audio = tmp_path / "talk.mp3"
    result = check_source(f'"{audio}"')
    assert not result.ok  # Play waits for the worker's file check
    assert result.needs_file_check
    assert not result.is_youtube
    assert result.error is None
    assert result.source == str(audio)


def test_check_source_never_touches_the_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object) -> bool:
        raise AssertionError("check_source must not touch the filesystem")

    monkeypatch.setattr("os.path.isfile", forbidden)
    monkeypatch.setattr("os.path.exists", forbidden)
    monkeypatch.setattr("os.path.isdir", forbidden)
    for text in (r"\\server\share\talk.mp3", "//server/share/talk.mp3", r"Z:\talk.mp3"):
        result = check_source(text)
        assert result.needs_file_check
        assert not result.ok


def test_check_source_plain_word_is_not_a_path() -> None:
    result = check_source("hello")
    assert not result.ok
    assert not result.needs_file_check
    assert result.error is not None


def test_links_need_no_file_check() -> None:
    assert not check_source("https://youtu.be/abc").needs_file_check
