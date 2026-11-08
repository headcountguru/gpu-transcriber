"""Shared application settings and the per-user data directory."""

import os
from dataclasses import dataclass
from pathlib import Path

# Container image the rented machine runs, pinned by digest (built by
# .github/workflows/build-image.yml from tag image-0.2, commit 97ad5a5).
# Update it with every new image build.
IMAGE = (
    "ghcr.io/headcountguru/gpu-transcriber"
    "@sha256:e6b16848b4ce933f44ca106fd4da0182c3440d0c9d936e1bdc660f0483d89ed0"
)

APP_DATA_DIR_NAME = "GpuTranscriber"


@dataclass(frozen=True)
class Settings:
    """Tunable thresholds and defaults for renting and running a GPU machine."""

    image: str
    min_vram_gb: float = 16.0
    min_reliability: float = 0.97
    min_down_mbps: float = 200.0
    min_cuda: float = 12.4
    max_price_per_hour: float = 1.0
    prefer_datacenter: bool = True
    disk_gb: int = 40
    poll_interval_s: float = 10.0
    max_machine_retries: int = 2
    label_prefix: str = "gputx-"
    remote_job_dir: str = "/work/job"


def app_data_dir() -> Path:
    """Return ``%APPDATA%/GpuTranscriber``, creating it if it does not exist yet."""
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    path = root / APP_DATA_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path
