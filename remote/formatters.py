import os

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS format"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"

def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS,mmm for SRT"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_outputs(segments: list[dict], output_dir: str):
    """
    Writes three formats to the output_dir.
    segments: [{start, end, text, speaker}]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    txt_path = os.path.join(output_dir, "transcript.txt")
    ts_txt_path = os.path.join(output_dir, "transcript_timestamped.txt")
    srt_path = os.path.join(output_dir, "transcript.srt")
    
    # 1) transcript.txt - grouped by speaker block
    with open(txt_path, "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in segments:
            if seg['speaker'] != current_speaker:
                if current_speaker is not None:
                    f.write("\n\n")
                f.write(f"{seg['speaker']}:\n")
                current_speaker = seg['speaker']
            
            # The segments could be sentences, just space them out
            f.write(seg['text'].strip() + " ")

    # 2) transcript_timestamped.txt
    with open(ts_txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            ts = format_timestamp(seg['start'])
            f.write(f"{ts} {seg['speaker']}: {seg['text'].strip()}\n")

    # 3) transcript.srt
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_srt = format_srt_timestamp(seg['start'])
            end_srt = format_srt_timestamp(seg['end'])
            f.write(f"{i}\n")
            f.write(f"{start_srt} --> {end_srt}\n")
            f.write(f"{seg['speaker']}: {seg['text'].strip()}\n\n")
