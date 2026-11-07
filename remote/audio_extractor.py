import os
import subprocess
import tempfile
import imageio_ffmpeg


def extract_audio(input_file: str) -> str:
    """
    Extracts audio from video/audio file and converts to 16kHz mono WAV.
    Returns path to temporary WAV file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-y", "-i", input_file, "-ac", "1", "-ar", "16000", temp_path]

    try:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(f"FFmpeg error: {result.stderr.decode('utf-8', errors='replace')}")
        return temp_path
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
