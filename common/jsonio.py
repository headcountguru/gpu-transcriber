"""Atomic JSON file helpers (stdlib only)."""

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def write_json_atomic(path: Path, obj: object) -> None:
    """Write ``obj`` as JSON to ``path`` via a temp file in the same dir + ``os.replace``."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> object | None:
    """Return parsed JSON from ``path``, or None if missing or unparseable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("Could not read JSON from %s", path)
        return None
