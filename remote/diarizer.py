from pyannote.audio import Pipeline
import logging
import torch

def diarize_audio(audio_path: str, hf_token: str = None, device: str = "cuda:0") -> list[dict]:
    """
    Diarizes the audio using pyannote speaker-diarization-3.1 model.
    Returns a list of speaker turns:
    [{'speaker': str, 'start': float, 'end': float}]
    """
    logging.info("Loading pyannote speaker-diarization pipeline...")

    # Set HF_TOKEN env var so huggingface_hub picks it up automatically.
    # Avoid calling login() — it does a whoami validation round-trip that
    # raises on network errors or token issues before the model even loads.
    if hf_token:
        import os
        os.environ['HF_TOKEN'] = hf_token

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    except Exception as e:
        raise RuntimeError(
            "Failed to load pyannote pipeline. Check:\n"
            "1. Your HF token is valid\n"
            "2. You accepted terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. You accepted terms at https://huggingface.co/pyannote/segmentation-3.0"
        ) from e

    if pipeline is None:
        raise RuntimeError(
            "pyannote returned None — your token likely doesn't have access to the model.\n"
            "Accept the terms at:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  https://huggingface.co/pyannote/segmentation-3.0\n"
            "Then re-run."
        )

    # Send pipeline to specified GPU
    pipeline.to(torch.device(device))

    logging.info(f"Diarizing {audio_path}...")
    diarization = pipeline(audio_path)
    
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end
        })
        
    return turns
