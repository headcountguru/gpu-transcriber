"""Contract test for the container image the rented GPU machine runs.

Skipped unless ``GPU_TRANSCRIBER_TEST_IMAGE`` names an image and ``docker`` is on PATH. The GitHub
Actions workflow ``build-image.yml`` sets it to the freshly built image and gates the push on it.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from _pipcheck import blocking_pip_check_lines

IMAGE = os.environ.get("GPU_TRANSCRIBER_TEST_IMAGE", "")

ENABLED = bool(IMAGE) and shutil.which("docker") is not None
# CI sets CI=true. There a missing image or docker must fail, never skip: an all-skipped run exits 0
# and would let an untested image through the build workflow's gate.
IN_CI = bool(os.environ.get("CI"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not ENABLED and not IN_CI,
        reason="set GPU_TRANSCRIBER_TEST_IMAGE and install docker to run the image contract",
    ),
]


@pytest.fixture(autouse=True)
def _require_image_when_in_ci():
    # Skip marks are evaluated before fixtures, so this only runs in CI (or when enabled).
    if not ENABLED:
        pytest.fail("CI requires GPU_TRANSCRIBER_TEST_IMAGE to name an image and docker on PATH")


# Runs `python -m pip <args>` inside the image and exits with pip's return code.
_PIP = "import subprocess, sys; sys.exit(subprocess.call([sys.executable, '-m', 'pip'{args}]))"

# Walks the installed pyannote namespace package directories (audio, core, metrics, ...).
_PYANNOTE_DIRS = (
    "import importlib.util; dirs = importlib.util.find_spec('pyannote').submodule_search_locations"
)


def run_in_image(image: str, code: str) -> subprocess.CompletedProcess:
    """Run ``python -c <code>`` in a throwaway container of ``image`` and capture the output."""
    return subprocess.run(
        ["docker", "run", "--rm", image, "python", "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_image_environment_matches_the_contract():
    code = (
        "import os, sys\n"
        "assert os.environ['PYTHONPATH'] == '/app:/app/remote', os.environ.get('PYTHONPATH')\n"
        "assert os.environ['HF_HOME'] == '/opt/hf', os.environ.get('HF_HOME')\n"
        "assert os.path.isfile('/app/remote/run_job.py')\n"
        "assert os.path.isfile('/app/remote/watchdog.py')\n"
        "assert os.path.isdir('/app/common')\n"
    )
    _assert_ok(run_in_image(IMAGE, code))


def test_pipeline_modules_import():
    _assert_ok(
        run_in_image(IMAGE, "import faster_whisper, pyannote.audio, common.protocol, run_job")
    )


def test_torchaudio_imports():
    _assert_ok(run_in_image(IMAGE, "import torchaudio"))


def test_installed_packages_are_consistent():
    result = run_in_image(IMAGE, _PIP.format(args=", 'check'"))
    problems = blocking_pip_check_lines(f"{result.stdout}\n{result.stderr}")
    assert result.returncode in (0, 1), f"pip check crashed:\n{result.stderr}"
    assert not problems, "pip check found problems:\n" + "\n".join(problems)


def test_torch_stack_is_the_base_image_build():
    # pip once replaced these with a CUDA 13 build that rented machines' drivers may not support.
    code = (
        "import torch, torchaudio\n"
        "assert torch.__version__.split('+')[0] == '2.5.1', torch.__version__\n"
        "assert torchaudio.__version__.split('+')[0] == '2.5.1', torchaudio.__version__\n"
        "assert torch.version.cuda == '12.4', torch.version.cuda\n"
    )
    _assert_ok(run_in_image(IMAGE, code))


def test_torchvision_is_absent():
    code = (
        "import importlib.util, sys\n"
        "sys.exit(0 if importlib.util.find_spec('torchvision') is None else 1)\n"
    )
    _assert_ok(run_in_image(IMAGE, code))


def test_pypi_watchdog_is_not_installed():
    # The PyPI "watchdog" package would collide with our remote/watchdog.py.
    result = run_in_image(IMAGE, _PIP.format(args=", 'show', 'watchdog'"))
    assert result.returncode != 0, f"PyPI watchdog is installed:\n{result.stdout}"


def test_vastai_cli_is_installed():
    _assert_ok(run_in_image(IMAGE, _PIP.format(args=", 'show', 'vastai'")))


def test_sshd_is_installed():
    _assert_ok(
        run_in_image(
            IMAGE, "import os, sys; sys.exit(0 if os.path.exists('/usr/sbin/sshd') else 1)"
        )
    )


def test_whisper_weights_are_baked_in():
    code = (
        "import pathlib, sys\n"
        "found = [p for p in pathlib.Path('/opt/hf').rglob('*') if 'large-v3-turbo' in p.name]\n"
        "sys.exit(0 if found else 1)\n"
    )
    _assert_ok(run_in_image(IMAGE, code))


def test_no_pyannote_model_is_baked_in():
    # pyannote models are gated; they are downloaded at job start with the user's own token.
    code = (
        "import pathlib, sys\n"
        "paths = pathlib.Path('/opt/hf').rglob('*')\n"
        "bad = [str(p) for p in paths if 'pyannote' in str(p).lower()]\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = run_in_image(IMAGE, code)
    assert result.returncode == 0, f"pyannote files under /opt/hf: {result.stdout}"


def test_pyannote_source_has_no_use_auth_token():
    code = (
        f"{_PYANNOTE_DIRS}\n"
        "import pathlib, sys\n"
        "bad = [\n"
        "    str(p)\n"
        "    for d in dirs\n"
        "    for p in pathlib.Path(d).rglob('*.py')\n"
        "    if 'use_auth_token' in p.read_text(encoding='utf-8', errors='ignore')\n"
        "]\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = run_in_image(IMAGE, code)
    assert result.returncode == 0, f"use_auth_token still present in: {result.stdout}"
