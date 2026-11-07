def test_formatters_write_the_three_output_files(tmp_path):
    from formatters import write_outputs

    write_outputs(
        [{"start": 0.0, "end": 1.5, "text": "hi", "speaker": "SPEAKER_00"}], str(tmp_path)
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "transcript.srt",
        "transcript.txt",
        "transcript_timestamped.txt",
    ]
    assert "[00:00:00] SPEAKER_00: hi" in (tmp_path / "transcript_timestamped.txt").read_text()
