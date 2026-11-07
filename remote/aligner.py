def align_words_with_speakers(words: list[dict], speaker_turns: list[dict]) -> list[dict]:
    """
    Takes word-level timestamps and speaker turns.
    Returns unified segments: [{start, end, text, speaker}]

    Speaker assignment uses majority-overlap: the speaker whose turn overlaps
    most with each word's time span wins. Falls back to nearest turn for
    words that fall in silence gaps between turns.
    """
    if not words:
        return []

    def get_speaker_for_word(word: dict) -> str:
        w_start, w_end = word['start'], word['end']
        w_dur = w_end - w_start

        if w_dur > 0:
            # Sum overlap seconds per speaker across all turns that touch this word
            overlap_by_speaker: dict[str, float] = {}
            for turn in speaker_turns:
                overlap = max(0.0, min(w_end, turn['end']) - max(w_start, turn['start']))
                if overlap > 0:
                    spk = turn['speaker']
                    overlap_by_speaker[spk] = overlap_by_speaker.get(spk, 0.0) + overlap
            if overlap_by_speaker:
                return max(overlap_by_speaker, key=overlap_by_speaker.get)
        else:
            # Zero-duration word — use midpoint containment
            t = w_start
            matches = [s for s in speaker_turns if s['start'] <= t <= s['end']]
            if matches:
                return matches[0]['speaker']

        # No overlap found — snap to nearest turn boundary
        if speaker_turns:
            mid = (w_start + w_end) / 2.0
            return min(
                speaker_turns,
                key=lambda s: min(abs(s['start'] - mid), abs(s['end'] - mid))
            )['speaker']

        return "UNKNOWN"

    # Assign each word its speaker
    for word in words:
        word['speaker'] = get_speaker_for_word(word)

    # Group consecutive same-speaker words into segments, splitting on speaker
    # change or silence gaps >= 2 seconds
    segments = []
    current = {
        'start': words[0]['start'],
        'end':   words[0]['end'],
        'text':  words[0]['word'],
        'speaker': words[0]['speaker'],
    }

    for word in words[1:]:
        same_speaker = word['speaker'] == current['speaker']
        small_gap    = (word['start'] - current['end']) < 2.0
        if same_speaker and small_gap:
            current['text'] += word['word']  # faster-whisper words carry leading space
            current['end']   = word['end']
        else:
            segments.append(current)
            current = {
                'start':   word['start'],
                'end':     word['end'],
                'text':    word['word'].lstrip(),
                'speaker': word['speaker'],
            }

    segments.append(current)
    return segments
