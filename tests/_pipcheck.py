"""Helper for the image contract test: interpret ``pip check`` output."""

from __future__ import annotations

import re

# The conda-built base image ships ninja metadata that pip cannot match to this platform, so
# `pip check` always prints this line and exits 1. ninja is only a build tool; nothing here uses it.
# Only this exact note for ninja is ignored; every other line still counts as a real problem.
_KNOWN_BASE_IMAGE_NOISE = re.compile(r"^ninja \S+ is not supported on this platform$")


def blocking_pip_check_lines(output: str) -> list[str]:
    """Return the ``pip check`` lines that indicate a real problem (empty means consistent)."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return [
        line
        for line in lines
        if line != "No broken requirements found." and not _KNOWN_BASE_IMAGE_NOISE.match(line)
    ]
