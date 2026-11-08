"""Shared pytest configuration for gpu-transcriber.

Import paths (repo root and ``remote/``) are configured in ``pyproject.toml``.
Qt tests run without a display: the offscreen platform is chosen here, before any Qt import,
unless the caller already picked one.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
