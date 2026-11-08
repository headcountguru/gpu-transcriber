"""Tests for preparing audio on the user's PC (compress, probe, YouTube download)."""

import math
import struct
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import pytest

from app.media_prep import (
    DOWNLOAD_TIMEOUT_S,
    MediaPrepError,
    PreparedAudio,
    compress_args,
    prepare_file,
    prepare_youtube,
    probe_duration_s,
    safe_folder_name,
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000
VIDEO_URL = "https://www.youtube.com/watch?v=abc123XYZ_-"
ILLEGAL = '<>:"/\\|?*'


def make_wav(path: Path, seconds: float = 2.0) -> Path:
    frames = int(SAMPLE_RATE * seconds)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)))
                for i in range(frames)
            )
        )
    return path


def make_silent_video(path: Path) -> Path:
    command = [
        FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-c:v", "mpeg4", str(path),
    ]  # fmt: skip
    subprocess.run(command, capture_output=True, check=True)
    return path


class FakeYtDlp:
    """A fake ``subprocess.run``: answers yt-dlp commands itself, sends ffmpeg to the real one."""

    def __init__(
        self,
        ytdlp_exe: Path,
        results: list["subprocess.CompletedProcess[str] | Exception"],
        write_wav_to: Path | None = None,
    ) -> None:
        self.ytdlp_exe = str(ytdlp_exe)
        self.results = list(results)
        self.write_wav_to = write_wav_to
        self.ytdlp_calls: list[list[str]] = []
        self.ytdlp_kwargs: list[dict[str, object]] = []

    def __call__(self, args: list[str], **kwargs: object) -> "subprocess.CompletedProcess[str]":
        command = [str(a) for a in args]
        if command[0] != self.ytdlp_exe:
            return subprocess.run(args, **kwargs)  # type: ignore[call-overload]
        self.ytdlp_calls.append(command)
        self.ytdlp_kwargs.append(kwargs)
        if "-U" in command:
            return subprocess.CompletedProcess(args, 0, "", "")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        if result.returncode in (0, 101) and self.write_wav_to is not None:
            make_wav(self.write_wav_to)
        return result


def completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> "subprocess.CompletedProcess[str]":
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@pytest.fixture
def ytdlp_exe(tmp_path: Path) -> Path:
    return tmp_path / "bin" / "yt-dlp.exe"


def test_compress_args_are_mono_16k_opus_32k() -> None:
    args = compress_args(Path("in.mp3"), Path("out.opus"))

    for pair in (["-ac", "1"], ["-ar", "16000"], ["-c:a", "libopus"], ["-b:a", "32k"]):
        index = args.index(pair[0])
        assert args[index + 1] == pair[1]
    assert args[args.index("-i") + 1] == "in.mp3"
    assert args[-1] == "out.opus"


def test_prepare_file_makes_small_opus_with_duration(tmp_path: Path) -> None:
    wav = make_wav(tmp_path / "talk.wav", 2.0)
    work = tmp_path / "work"

    prepared = prepare_file(wav, work, FFMPEG)

    assert isinstance(prepared, PreparedAudio)
    assert prepared.path == work / "audio.opus"
    assert prepared.path.is_file()
    assert prepared.title == "talk"
    assert prepared.duration_s == pytest.approx(2.0, abs=0.2)
    assert prepared.size_bytes == prepared.path.stat().st_size
    assert prepared.size_bytes < wav.stat().st_size


def test_prepare_file_rejects_file_without_audio(tmp_path: Path) -> None:
    video = make_silent_video(tmp_path / "silent.mp4")

    with pytest.raises(MediaPrepError) as caught:
        prepare_file(video, tmp_path / "work", FFMPEG)

    assert "no audio" in str(caught.value).lower()
    assert not (tmp_path / "work" / "audio.opus").exists()


def test_prepare_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MediaPrepError) as caught:
        prepare_file(tmp_path / "nope.mp3", tmp_path / "work", FFMPEG)

    assert "find" in str(caught.value).lower()


def test_prepare_file_rejects_zero_byte_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp3"
    empty.write_bytes(b"")

    with pytest.raises(MediaPrepError) as caught:
        prepare_file(empty, tmp_path / "work", FFMPEG)

    assert "empty" in str(caught.value).lower()


def test_prepare_file_rejects_unreadable_file_in_plain_words(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp3"
    junk.write_text("this is not audio", encoding="utf-8")

    with pytest.raises(MediaPrepError) as caught:
        prepare_file(junk, tmp_path / "work", FFMPEG)

    message = str(caught.value)
    assert "can't read" in message.lower()
    assert "Traceback" not in message
    assert "ffmpeg" not in message.lower()
    assert "Invalid data" not in message


def test_prepare_youtube_reports_plain_error_for_unavailable_video(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    stderr = "ERROR: [youtube] abc: Video unavailable\nTraceback (most recent call last):\n"
    run = FakeYtDlp(ytdlp_exe, [completed(1, "", stderr), completed(1, "", stderr)])

    with pytest.raises(MediaPrepError) as caught:
        prepare_youtube(VIDEO_URL, tmp_path / "work", FFMPEG, ytdlp_exe, run=run)

    message = str(caught.value)
    assert "unavailable" in message.lower()
    assert "Traceback" not in message
    assert "ERROR:" not in message


def test_prepare_youtube_rejects_playlist_url(tmp_path: Path, ytdlp_exe: Path) -> None:
    run = FakeYtDlp(ytdlp_exe, [])

    with pytest.raises(MediaPrepError) as caught:
        prepare_youtube(
            "https://www.youtube.com/playlist?list=PL123", tmp_path, FFMPEG, ytdlp_exe, run=run
        )

    assert "playlists are not supported" in str(caught.value).lower()
    assert run.ytdlp_calls == []


def test_prepare_youtube_accepts_a_video_link_that_also_names_a_playlist(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    work = tmp_path / "work"
    downloaded = work / "download.wav"
    run = FakeYtDlp(ytdlp_exe, [completed(0, f"A talk\n{downloaded}\n")], write_wav_to=downloaded)

    prepared = prepare_youtube(VIDEO_URL + "&list=PL123", work, FFMPEG, ytdlp_exe, run=run)

    assert prepared.title == "A talk"


def test_prepare_youtube_rejects_something_that_is_not_a_web_link(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    run = FakeYtDlp(ytdlp_exe, [])

    with pytest.raises(MediaPrepError) as caught:
        prepare_youtube("not a link", tmp_path, FFMPEG, ytdlp_exe, run=run)

    assert "link" in str(caught.value).lower()
    assert run.ytdlp_calls == []


def test_prepare_youtube_downloads_and_compresses(tmp_path: Path, ytdlp_exe: Path) -> None:
    work = tmp_path / "work"
    downloaded = work / "download.wav"
    run = FakeYtDlp(
        ytdlp_exe, [completed(0, f"My talk: part 1\n{downloaded}\n")], write_wav_to=downloaded
    )

    prepared = prepare_youtube(VIDEO_URL, work, FFMPEG, ytdlp_exe, run=run)

    assert prepared.path == work / "audio.opus"
    assert prepared.title == "My talk: part 1"
    assert prepared.duration_s == pytest.approx(2.0, abs=0.2)
    command = run.ytdlp_calls[0]
    assert "--no-playlist" in command
    assert command[command.index("-f") + 1] == "bestaudio"
    assert "before_dl:%(title)s" in command
    assert "after_move:filepath" in command
    assert command[-1] == VIDEO_URL
    assert "--cookies" not in command
    assert not downloaded.exists()


def test_prepare_youtube_passes_the_cookies_file(tmp_path: Path, ytdlp_exe: Path) -> None:
    work = tmp_path / "work"
    downloaded = work / "download.wav"
    cookies = tmp_path / "cookies.txt"
    run = FakeYtDlp(ytdlp_exe, [completed(0, f"T\n{downloaded}\n")], write_wav_to=downloaded)

    prepare_youtube(VIDEO_URL, work, FFMPEG, ytdlp_exe, cookies_file=cookies, run=run)

    command = run.ytdlp_calls[0]
    assert command[command.index("--cookies") + 1] == str(cookies)


def test_prepare_youtube_updates_and_retries_once_after_a_failure(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    work = tmp_path / "work"
    downloaded = work / "download.wav"
    run = FakeYtDlp(
        ytdlp_exe,
        [completed(1, "", "ERROR: HTTP Error 403"), completed(0, f"T\n{downloaded}\n")],
        write_wav_to=downloaded,
    )

    prepared = prepare_youtube(VIDEO_URL, work, FFMPEG, ytdlp_exe, run=run)

    assert prepared.title == "T"
    kinds = ["update" if "-U" in call else "download" for call in run.ytdlp_calls]
    assert kinds == ["download", "update", "download"]


def test_prepare_youtube_gives_a_plain_error_for_a_live_stream(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    stderr = "ERROR: [youtube] abc: This live event will begin in 2 hours."
    run = FakeYtDlp(ytdlp_exe, [completed(1, "", stderr), completed(1, "", stderr)])

    with pytest.raises(MediaPrepError) as caught:
        prepare_youtube(VIDEO_URL, tmp_path / "work", FFMPEG, ytdlp_exe, run=run)

    assert "live" in str(caught.value).lower()


def test_prepare_youtube_never_leaks_cookie_text_in_errors(tmp_path: Path, ytdlp_exe: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text(".youtube.com\tTRUE\t/\tTRUE\t0\tSID\tSECRETVALUE", encoding="utf-8")
    stderr = "ERROR: something odd with SECRETVALUE"
    run = FakeYtDlp(ytdlp_exe, [completed(1, "", stderr), completed(1, "", stderr)])

    with pytest.raises(MediaPrepError) as caught:
        prepare_youtube(
            VIDEO_URL, tmp_path / "work", FFMPEG, ytdlp_exe, cookies_file=cookies, run=run
        )

    assert "SECRETVALUE" not in str(caught.value)


def test_prepare_youtube_empty_download_is_an_error(tmp_path: Path, ytdlp_exe: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    ghost = work / "download.m4a"
    run = FakeYtDlp(ytdlp_exe, [completed(0, f"T\n{ghost}\n")])

    with pytest.raises(MediaPrepError) as caught:
        prepare_youtube(VIDEO_URL, work, FFMPEG, ytdlp_exe, run=run)

    assert "download" in str(caught.value).lower()


def test_probe_duration_reads_file_length_without_converting(tmp_path: Path) -> None:
    wav = make_wav(tmp_path / "talk.wav", 2.0)

    seconds = probe_duration_s(str(wav), False, FFMPEG, tmp_path / "unused.exe")

    assert seconds == pytest.approx(2.0, abs=0.1)
    assert [p.name for p in tmp_path.iterdir()].count("audio.opus") == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["talk.wav"]


def test_probe_duration_rejects_file_without_audio(tmp_path: Path) -> None:
    video = make_silent_video(tmp_path / "silent.mp4")

    with pytest.raises(MediaPrepError) as caught:
        probe_duration_s(str(video), False, FFMPEG, tmp_path / "unused.exe")

    assert "no audio" in str(caught.value).lower()


def test_probe_duration_rejects_zero_byte_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp3"
    empty.write_bytes(b"")

    with pytest.raises(MediaPrepError):
        probe_duration_s(str(empty), False, FFMPEG, tmp_path / "unused.exe")


def test_probe_duration_for_youtube_parses_yt_dlp_output(tmp_path: Path, ytdlp_exe: Path) -> None:
    run = FakeYtDlp(ytdlp_exe, [completed(0, "3725\n")])

    seconds = probe_duration_s(VIDEO_URL, True, FFMPEG, ytdlp_exe, run=run)

    assert seconds == 3725.0
    command = run.ytdlp_calls[0]
    assert "--skip-download" in command
    assert command[command.index("--print") + 1] == "duration"
    assert "--no-playlist" in command


def test_probe_duration_for_youtube_rejects_playlists(tmp_path: Path, ytdlp_exe: Path) -> None:
    run = FakeYtDlp(ytdlp_exe, [])

    with pytest.raises(MediaPrepError) as caught:
        probe_duration_s(
            "https://www.youtube.com/playlist?list=PL1", True, FFMPEG, ytdlp_exe, run=run
        )

    assert "playlists are not supported" in str(caught.value).lower()


def test_probe_duration_for_a_live_stream_is_a_plain_error(tmp_path: Path, ytdlp_exe: Path) -> None:
    run = FakeYtDlp(ytdlp_exe, [completed(0, "NA\n")])

    with pytest.raises(MediaPrepError) as caught:
        probe_duration_s(VIDEO_URL, True, FFMPEG, ytdlp_exe, run=run)

    assert "live" in str(caught.value).lower()


def test_probe_duration_for_unavailable_video_is_a_plain_error(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    stderr = "ERROR: [youtube] abc: Private video. Sign in if you've been granted access"
    run = FakeYtDlp(ytdlp_exe, [completed(1, "", stderr), completed(1, "", stderr)])

    with pytest.raises(MediaPrepError) as caught:
        probe_duration_s(VIDEO_URL, True, FFMPEG, ytdlp_exe, run=run)

    assert "unavailable" in str(caught.value).lower()


def test_safe_folder_name_strips_windows_illegal_characters() -> None:
    cleaned = safe_folder_name('a:b?c/d\\e|f*g<h>i"j\x07k')

    assert not any(ch in cleaned for ch in ILLEGAL)
    assert all(ord(ch) >= 32 for ch in cleaned)
    assert cleaned == "abcdefghijk"


def test_safe_folder_name_handles_reserved_long_and_empty() -> None:
    assert safe_folder_name("CON") != "CON"
    assert safe_folder_name("nul.txt").lower() != "nul.txt"
    assert safe_folder_name("com1") != "com1"
    assert len(safe_folder_name("x" * 300)) <= 80
    assert safe_folder_name("") == "untitled"
    assert safe_folder_name("???") == "untitled"
    trailing = safe_folder_name("name.. ")
    assert trailing == "name"
    assert not trailing.endswith((".", " "))


def test_safe_folder_name_cap_does_not_leave_trailing_space() -> None:
    cleaned = safe_folder_name("x" * 79 + " tail")

    assert len(cleaned) <= 80
    assert not cleaned.endswith((" ", "."))


def test_safe_folder_name_keeps_ordinary_titles() -> None:
    assert safe_folder_name("Weekly meeting - 2026") == "Weekly meeting - 2026"


@pytest.mark.parametrize(
    "link",
    [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtube.com/watch?v=abc123&t=30s",
        "https://m.youtube.com/watch?v=abc123",
        "https://music.youtube.com/watch?v=abc123",
        "https://youtu.be/abc123",
        "https://youtu.be/abc123?list=PL1",
        "https://www.youtube.com/shorts/abc123",
        "https://www.youtube.com/live/abc123",
        "https://www.youtube.com/embed/abc123",
        "https://www.youtube.com/embed/abc123?list=PL1",
    ],
)
def test_probe_accepts_single_video_link_shapes(link: str, ytdlp_exe: Path) -> None:
    run = FakeYtDlp(ytdlp_exe, [completed(0, "60\n")])

    assert probe_duration_s(link, True, FFMPEG, ytdlp_exe, run=run) == 60.0


@pytest.mark.parametrize(
    "link",
    [
        "https://www.youtube.com/@somechannel",
        "https://www.youtube.com/@somechannel/videos",
        "https://www.youtube.com/channel/UC123",
        "https://www.youtube.com/user/someone",
        "https://www.youtube.com/c/somename",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?v=",
        "https://www.youtube.com/shorts/",
        "https://youtu.be/",
        "https://notyoutu.be/abc123",
        "https://www.youtube.com.evil.example/watch?v=abc123",
        "https://example.com/watch?v=abc123",
        "http://[bad",
    ],
)
def test_prepare_youtube_rejects_links_that_are_not_a_single_video(
    link: str, tmp_path: Path, ytdlp_exe: Path
) -> None:
    run = FakeYtDlp(ytdlp_exe, [])

    with pytest.raises(MediaPrepError) as caught:
        prepare_youtube(link, tmp_path, FFMPEG, ytdlp_exe, run=run)

    message = str(caught.value)
    assert "single video" in message.lower()
    assert "Traceback" not in message
    assert run.ytdlp_calls == []


def test_probe_rejects_links_that_are_not_a_single_video(tmp_path: Path, ytdlp_exe: Path) -> None:
    run = FakeYtDlp(ytdlp_exe, [])

    with pytest.raises(MediaPrepError) as caught:
        probe_duration_s("https://www.youtube.com/@somechannel", True, FFMPEG, ytdlp_exe, run=run)

    assert "single video" in str(caught.value).lower()
    assert run.ytdlp_calls == []


def test_prepare_youtube_refuses_live_streams_and_stops_after_one_download(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    work = tmp_path / "work"
    downloaded = work / "download.wav"
    run = FakeYtDlp(ytdlp_exe, [completed(0, f"T\n{downloaded}\n")], write_wav_to=downloaded)

    prepare_youtube(VIDEO_URL, work, FFMPEG, ytdlp_exe, run=run)

    command = run.ytdlp_calls[0]
    assert command[command.index("--match-filter") + 1] == "!is_live"
    assert command[command.index("--max-downloads") + 1] == "1"


def test_prepare_youtube_says_live_stream_when_the_filter_skips_the_video(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    run = FakeYtDlp(ytdlp_exe, [completed(0, "", "")])

    with pytest.raises(MediaPrepError) as caught:
        prepare_youtube(VIDEO_URL, tmp_path / "work", FFMPEG, ytdlp_exe, run=run)

    assert "live stream" in str(caught.value).lower()
    assert "empty" not in str(caught.value).lower()
    assert run.ytdlp_calls[1:] == []


def test_prepare_youtube_treats_max_downloads_exit_code_as_success(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    work = tmp_path / "work"
    downloaded = work / "download.wav"
    run = FakeYtDlp(ytdlp_exe, [completed(101, f"T\n{downloaded}\n")], write_wav_to=downloaded)

    prepared = prepare_youtube(VIDEO_URL, work, FFMPEG, ytdlp_exe, run=run)

    assert prepared.title == "T"
    assert all("-U" not in call for call in run.ytdlp_calls)


def test_prepare_youtube_download_has_a_generous_finite_timeout(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    work = tmp_path / "work"
    downloaded = work / "download.wav"
    run = FakeYtDlp(ytdlp_exe, [completed(0, f"T\n{downloaded}\n")], write_wav_to=downloaded)

    prepare_youtube(VIDEO_URL, work, FFMPEG, ytdlp_exe, run=run)

    assert run.ytdlp_kwargs[0]["timeout"] == DOWNLOAD_TIMEOUT_S
    assert 3600 <= DOWNLOAD_TIMEOUT_S <= 24 * 3600


def test_prepare_youtube_turns_a_timeout_into_a_plain_error(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    run = FakeYtDlp(ytdlp_exe, [subprocess.TimeoutExpired(["yt-dlp"], DOWNLOAD_TIMEOUT_S)])

    with pytest.raises(MediaPrepError) as caught:
        prepare_youtube(VIDEO_URL, tmp_path / "work", FFMPEG, ytdlp_exe, run=run)

    message = str(caught.value)
    assert "too long" in message.lower()
    assert "TimeoutExpired" not in message


def test_prepare_youtube_runs_yt_dlp_with_utf8_output(tmp_path: Path, ytdlp_exe: Path) -> None:
    work = tmp_path / "work"
    downloaded = work / "download.wav"
    run = FakeYtDlp(ytdlp_exe, [completed(0, f"T\n{downloaded}\n")], write_wav_to=downloaded)

    prepare_youtube(VIDEO_URL, work, FFMPEG, ytdlp_exe, run=run)

    env = run.ytdlp_kwargs[0]["env"]
    assert isinstance(env, dict)
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert run.ytdlp_kwargs[0]["encoding"] == "utf-8"


def test_prepare_youtube_finds_the_file_when_the_printed_path_is_garbled(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    work = tmp_path / "José" / "work"
    downloaded = work / "download.wav"
    garbled = str(downloaded).replace("é", "��")
    run = FakeYtDlp(ytdlp_exe, [completed(0, f"Título\n{garbled}\n")], write_wav_to=downloaded)

    prepared = prepare_youtube(VIDEO_URL, work, FFMPEG, ytdlp_exe, run=run)

    assert prepared.title == "Título"
    assert prepared.duration_s == pytest.approx(2.0, abs=0.2)


def test_prepare_youtube_clears_stale_downloads_and_ignores_partial_files(
    tmp_path: Path, ytdlp_exe: Path
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    stale = work / "download.m4a"
    stale.write_bytes(b"old junk from an earlier run")
    downloaded = work / "download.wav"
    seen: list[bool] = []

    class Checking(FakeYtDlp):
        def __call__(self, args, **kwargs):  # type: ignore[no-untyped-def]
            if str(args[0]) == self.ytdlp_exe and "-U" not in args:
                seen.append(stale.exists())
                (work / "download.wav.part").write_bytes(b"half")
            return super().__call__(args, **kwargs)

    run = Checking(ytdlp_exe, [completed(0, f"T\n{downloaded}\n")], write_wav_to=downloaded)

    prepared = prepare_youtube(VIDEO_URL, work, FFMPEG, ytdlp_exe, run=run)

    assert seen == [False]
    assert prepared.duration_s == pytest.approx(2.0, abs=0.2)


def test_output_folder_problem_is_not_reported_as_a_missing_input_file(tmp_path: Path) -> None:
    # Seen live: ffmpeg could not write into the app's working folder and said "No such file or
    # directory" about its OUTPUT, which used to be shown as "Can't find that file" (the input).
    source = make_wav(tmp_path / "in.wav")

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if "-c:a" in args:  # the conversion step
            return subprocess.CompletedProcess(
                args, 4294967294, "", f"{args[-1]}: No such file or directory\n"
            )
        return subprocess.run(args, **kwargs)  # type: ignore[call-overload]  # the real probe

    with pytest.raises(MediaPrepError) as caught:
        prepare_file(source, tmp_path / "work", FFMPEG, run=run)

    message = str(caught.value)
    assert "find that file" not in message
    assert "working folder" in message
