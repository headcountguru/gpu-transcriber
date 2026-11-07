"""
Optional forced alignment using whisperx (wav2vec2/CTC).

Refines word-level timestamps produced by faster-whisper, giving phoneme-accurate
boundaries. Silently falls back to the original timestamps if whisperx is not
installed or alignment fails.
"""
import logging


def apply_forced_alignment(
    words: list[dict],
    audio_path: str,
    language: str,
    device: str = "cuda",
) -> list[dict]:
    """
    Attempt to refine word timestamps with whisperx forced alignment.

    Returns the original `words` list unchanged if whisperx is unavailable
    or alignment raises an error.
    """
    try:
        import whisperx  # noqa: F401 — presence check
    except ImportError:
        logging.info("whisperx not installed — skipping forced alignment.")
        return words

    try:
        import soundfile as sf

        audio_arr, sr = sf.read(audio_path, dtype="float32")
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=1)

        segments = _words_to_segments(words)
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(
            segments, model_a, metadata, audio_arr, device,
            return_char_alignments=False,
        )

        aligned = result.get("word_segments", [])
        if not aligned:
            logging.warning("Forced alignment returned empty result — keeping original timestamps.")
            return words

        refined = [
            {
                "word":  w.get("word", ""),
                "start": w.get("start", 0.0),
                "end":   w.get("end",   0.0),
            }
            for w in aligned
        ]
        logging.info(f"Forced alignment complete — {len(refined)} words refined.")
        return refined

    except Exception as exc:
        logging.warning(f"Forced alignment failed ({exc}) — keeping original timestamps.")
        return words


# ── helpers ──────────────────────────────────────────────────────────────────

def _words_to_segments(words: list[dict]) -> list[dict]:
    """
    Group words into coarse segments (split on >2 s silences) for whisperx.
    whisperx.align() expects [{start, end, text, words: [{word, start, end}]}].
    """
    if not words:
        return []

    segments: list[list[dict]] = []
    bucket = [words[0]]
    for w in words[1:]:
        if w["start"] - bucket[-1]["end"] > 2.0:
            segments.append(bucket)
            bucket = []
        bucket.append(w)
    if bucket:
        segments.append(bucket)

    return [_make_segment(g) for g in segments]


def _make_segment(group: list[dict]) -> dict:
    text = "".join(w["word"] for w in group).strip()
    return {
        "start": group[0]["start"],
        "end":   group[-1]["end"],
        "text":  text,
        "words": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in group],
    }
