import os
import tempfile
import logging
import soundfile as sf

from diarizer import diarize_audio

# Files longer than this get chunked before diarization
CHUNK_THRESHOLD_MINUTES = 60
CHUNK_MINUTES           = 30
OVERLAP_MINUTES         = 2


def chunk_and_diarize(audio_path: str, hf_token: str, device: str,
                      progress_callback=None) -> list[dict]:
    """
    Diarizes audio, chunking into overlapping segments for long files.

    Files <= CHUNK_THRESHOLD_MINUTES are passed directly to diarize_audio().
    Longer files are split into CHUNK_MINUTES chunks with OVERLAP_MINUTES
    overlap.  Speaker labels are reconciled across chunk boundaries using
    coverage in the overlap region, then turns are merged into a single
    timeline.

    Returns: [{'speaker': str, 'start': float, 'end': float}, ...]
    """
    with sf.SoundFile(audio_path) as f:
        sr          = f.samplerate
        total_secs  = len(f) / sr

    threshold = CHUNK_THRESHOLD_MINUTES * 60
    if total_secs <= threshold:
        return diarize_audio(audio_path, hf_token=hf_token, device=device)

    chunk_secs   = CHUNK_MINUTES   * 60
    overlap_secs = OVERLAP_MINUTES * 60

    # ── Build chunk windows ───────────────────────────────────────────────────
    windows = []
    start = 0.0
    while start < total_secs:
        end = min(start + chunk_secs, total_secs)
        windows.append((start, end))
        if end >= total_secs:
            break
        start = end - overlap_secs

    n = len(windows)
    logging.info(
        f"Audio {total_secs/60:.1f} min — chunking into {n} x {CHUNK_MINUTES}-min "
        f"segments (overlap {OVERLAP_MINUTES} min)"
    )

    # ── Diarize each chunk ────────────────────────────────────────────────────
    chunk_turns = []   # list of (chunk_start, chunk_end, [abs-time turns])
    tmp_files   = []

    try:
        for i, (win_start, win_end) in enumerate(windows):
            if progress_callback:
                progress_callback(
                    "diarization", 65,
                    f"Diarizing chunk {i+1}/{n}  "
                    f"({_fmt(win_start)} – {_fmt(win_end)})  |  "
                    f"{n - i - 1} chunk(s) remaining after this"
                )

            tmp = _extract_chunk(audio_path, win_start, win_end, sr)
            tmp_files.append(tmp)

            # Turns returned are relative to the chunk start
            rel_turns = diarize_audio(tmp, hf_token=hf_token, device=device)

            # Shift to absolute timestamps
            abs_turns = [
                {'speaker': t['speaker'],
                 'start':   t['start'] + win_start,
                 'end':     t['end']   + win_start}
                for t in rel_turns
            ]
            chunk_turns.append((win_start, win_end, abs_turns))

    finally:
        for tmp in tmp_files:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ── Reconcile speaker labels and merge ────────────────────────────────────
    return _reconcile_and_merge(chunk_turns, overlap_secs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_chunk(audio_path: str, start_sec: float, end_sec: float,
                   sr: int) -> str:
    """Read a time slice and write it to a temp WAV file."""
    with sf.SoundFile(audio_path) as f:
        f.seek(int(start_sec * sr))
        data = f.read(int((end_sec - start_sec) * sr))
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    sf.write(path, data, sr)
    return path


def _coverage_in_window(turns: list[dict], t_start: float,
                        t_end: float) -> dict[str, float]:
    """Return {speaker: seconds} for turns overlapping [t_start, t_end]."""
    cov: dict[str, float] = {}
    for t in turns:
        seg_s = max(t['start'], t_start)
        seg_e = min(t['end'],   t_end)
        if seg_e > seg_s:
            cov[t['speaker']] = cov.get(t['speaker'], 0.0) + (seg_e - seg_s)
    return cov


def _match_speakers(cov_curr: dict[str, float],
                    cov_prev: dict[str, float]) -> dict[str, str]:
    """
    Greedy match: map each speaker in cov_curr to the best-matching speaker
    in cov_prev using a Jaccard-like overlap score.
    Returns {curr_speaker: prev_speaker}.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for curr_spk in sorted(cov_curr, key=lambda s: -cov_curr[s]):
        best_prev, best_score = None, 0.0
        for prev_spk, prev_cov in cov_prev.items():
            if prev_spk in used:
                continue
            score = min(cov_curr[curr_spk], prev_cov) / max(cov_curr[curr_spk], prev_cov)
            if score > best_score:
                best_score, best_prev = score, prev_spk
        if best_prev and best_score >= 0.25:
            mapping[curr_spk] = best_prev
            used.add(best_prev)
    return mapping


def _reconcile_and_merge(chunk_turns: list[tuple], overlap_secs: float) -> list[dict]:
    """
    Build a global speaker label mapping across chunks using the overlap
    region, then merge all turns into a single sorted timeline, excluding
    duplicate coverage in overlap zones.
    """
    if not chunk_turns:
        return []

    # label_map[(chunk_idx, local_speaker)] = global_speaker_label
    label_map: dict[tuple, str] = {}

    # Seed with chunk 0's labels unchanged
    for t in chunk_turns[0][2]:
        label_map[(0, t['speaker'])] = t['speaker']

    global_speakers: set[str] = {v for v in label_map.values()}

    def _next_label(used: set[str]) -> str:
        idx = 0
        while f"SPEAKER_{idx:02d}" in used:
            idx += 1
        return f"SPEAKER_{idx:02d}"

    for i in range(1, len(chunk_turns)):
        chunk_start, _, turns_i   = chunk_turns[i]
        _,  prev_end, turns_prev  = chunk_turns[i - 1]

        # Overlap window in absolute time
        overlap_start = chunk_start
        overlap_end   = prev_end

        cov_prev = _coverage_in_window(turns_prev, overlap_start, overlap_end)
        cov_curr = _coverage_in_window(turns_i,    overlap_start, overlap_end)

        local_to_prev = _match_speakers(cov_curr, cov_prev)

        local_speakers = {t['speaker'] for t in turns_i}
        for local_spk in local_speakers:
            if local_spk in local_to_prev:
                prev_local  = local_to_prev[local_spk]
                global_lbl  = label_map.get((i - 1, prev_local), prev_local)
            else:
                global_lbl  = _next_label(global_speakers)
                global_speakers.add(global_lbl)
            label_map[(i, local_spk)] = global_lbl

        global_speakers.update(label_map.values())

    # ── Merge, excluding overlap zones already covered by the next chunk ─────
    merged: list[dict] = []

    for i, (chunk_start, chunk_end, turns_i) in enumerate(chunk_turns):
        # Keep only the non-overlap portion of each chunk.
        # First chunk: full content up to where next chunk begins.
        # Middle chunks: from where prev chunk ended (overlap start) to where
        #                next chunk begins.
        # Last chunk: from overlap start to end.
        cut_start = chunk_start if i == 0 else chunk_turns[i][0]
        cut_end   = chunk_turns[i + 1][0] if i < len(chunk_turns) - 1 else chunk_end

        for t in turns_i:
            seg_s = max(t['start'], cut_start)
            seg_e = min(t['end'],   cut_end)
            if seg_e <= seg_s:
                continue
            global_spk = label_map.get((i, t['speaker']), t['speaker'])
            merged.append({'speaker': global_spk, 'start': seg_s, 'end': seg_e})

    merged.sort(key=lambda x: x['start'])
    return merged


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}:{m:02d}"
