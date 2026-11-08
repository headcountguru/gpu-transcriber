"""The install self-check: ``python -m app.main --selftest`` exits 0 without a window."""

import os
import subprocess
import sys

import pytest

SELFTEST_TIMEOUT_S = 120


@pytest.mark.integration
def test_selftest_exits_zero() -> None:
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}

    result = subprocess.run(
        [sys.executable, "-m", "app.main", "--selftest"],
        env=env,
        capture_output=True,
        text=True,
        timeout=SELFTEST_TIMEOUT_S,
    )

    assert result.returncode == 0, result.stderr
