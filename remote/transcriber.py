import time
import torch
from faster_whisper import WhisperModel
import logging

def transcribe_audio(audio_path: str, model_size: str = "large-v3-turbo", language: str = None, device: str = "cuda:0", progress_callback=None) -> tuple[list[dict], float, str]:
    """
    Transcribes audio using faster-whisper.
    Returns: (word list, audio duration in seconds)
    """
    logging.info(f"Loading faster-whisper model: {model_size} on {device}")

    # CTranslate2 uses device="cuda" + device_index=N, not "cuda:N"
    if ':' in device:
        _dev, _idx = device.split(':')
        model = WhisperModel(model_size, device=_dev, device_index=int(_idx), compute_type="float16")
    else:
        model = WhisperModel(model_size, device=device, compute_type="float16")
    
    logging.info(f"Transcribing {audio_path}...")
    segments, info = model.transcribe(
        _load_audio(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    
    logging.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
    duration = info.duration
    
    words = []
    total_words = 0
    start_wall = time.time()
    for segment in segments:
        for word in segment.words:
            words.append({
                'word': word.word,
                'start': word.start,
                'end': word.end
            })
            total_words += 1
            if progress_callback and total_words % 100 == 0:
                progress_pct = int((word.end / duration) * 100)
                overall_pct = 30 + int(progress_pct * 0.3)
                elapsed_wall = time.time() - start_wall
                if elapsed_wall > 0 and word.end > 0:
                    speed = word.end / elapsed_wall          # audio-sec per wall-sec
                    remaining = (duration - word.end) / speed
                    suffix = f"est. remaining ~{format_time(remaining)}" if remaining > 0 else "almost done..."
                else:
                    suffix = "calculating..."
                progress_callback("transcription", overall_pct,
                                  f"Transcribing: {format_time(word.end)} / {format_time(duration)}  |  {suffix}")
            
    return words, duration, info.language

def _load_audio(audio_path: str):
    """
    Load the 16kHz mono WAV with soundfile and hand faster-whisper a numpy array,
    bypassing its PyAV decoder (older PyAV builds reject `metadata_errors`).
    Falls back to the path if the file isn't 16kHz.
    """
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        return audio_path
    return audio

def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"
