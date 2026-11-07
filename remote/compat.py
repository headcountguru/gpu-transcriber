"""Runtime compatibility patches for the pyannote / torch / torchaudio stack.

Ported from the Kaggle notebook. Every target is imported lazily and a target that
cannot be imported is skipped with a warning, so the patches are safe to call anywhere.
All patch functions are module level so repeated calls install the same objects.
The ``use_auth_token`` -> ``token`` source patch is done at image build time, not here.
"""

import importlib
import logging
from types import ModuleType

logger = logging.getLogger(__name__)

_IMPORT_ERRORS = (ImportError, OSError)


class AudioMetaData:
    """Stand-in for ``torchaudio.AudioMetaData`` (removed in newer torchaudio)."""


class _AudioInfo(AudioMetaData):
    """``torchaudio.info`` result built from ``soundfile.info``."""

    def __init__(self, path: str) -> None:
        import soundfile

        info = soundfile.info(path)
        self.sample_rate = info.samplerate
        self.num_frames = info.frames
        self.num_channels = info.channels
        self.bits_per_sample = 16
        self.encoding = "PCM_S"


def _noop_login(token: str | None = None, **kwargs: object) -> None:
    """Replacement for ``huggingface_hub.login``; auth comes from the HF_TOKEN env var."""


def _force_full_load(
    path_or_url: object, map_location: object = None, weights_only: object = None, **kwargs: object
) -> object:
    """Replacement for ``lightning_fabric`` ``_load``: torch 2.6 defaults to weights_only=True."""
    import torch

    return torch.load(path_or_url, map_location=map_location, weights_only=False)


def _audio_info(filepath: str, backend: object = None) -> AudioMetaData:
    return _AudioInfo(filepath)


def _list_audio_backends() -> list[str]:
    return ["soundfile"]


def _get_audio_backend() -> str:
    return "soundfile"


def _try_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except _IMPORT_ERRORS as exc:
        logger.warning("Compat patch skipped: cannot import %s (%s)", name, exc)
        return None


def _patch_huggingface_login() -> None:
    hub = _try_import("huggingface_hub")
    if hub is not None:
        hub.login = _noop_login


def _patch_lightning_load() -> None:
    cloud_io = _try_import("lightning_fabric.utilities.cloud_io")
    if cloud_io is None or _try_import("torch") is None:
        return
    cloud_io._load = _force_full_load


def _patch_torchaudio() -> None:
    torchaudio = _try_import("torchaudio")
    if torchaudio is None:
        return
    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = _list_audio_backends
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = _get_audio_backend
    if not hasattr(torchaudio, "info") and _try_import("soundfile") is not None:
        torchaudio.info = _audio_info
        logger.info("torchaudio.info: patched via soundfile")


def apply_compat_patches() -> None:
    """Apply all runtime patches. Idempotent; unavailable targets are skipped."""
    _patch_huggingface_login()
    _patch_lightning_load()
    _patch_torchaudio()
