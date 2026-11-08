"""Static checks on the image's build inputs (run locally, no Docker needed).

The first image build failed because pip replaced the base image's torch 2.5.1 (CUDA 12.4) with a
newer CUDA 13 build, and the newer torchaudio no longer has ``AudioMetaData``, which pyannote needs.
A constraints file keeps torch and torchaudio at the base image's versions.
"""

from __future__ import annotations

from pathlib import Path

from _pipcheck import blocking_pip_check_lines

DOCKER_DIR = Path(__file__).resolve().parent.parent / "docker"
BASE_TORCH_VERSION = "2.5.1"


def _constraint_lines() -> list[str]:
    lines = (DOCKER_DIR / "constraints.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def test_constraints_pin_torch_and_torchaudio_to_the_base_image_version():
    constraints = _constraint_lines()
    assert f"torch=={BASE_TORCH_VERSION}" in constraints
    assert f"torchaudio=={BASE_TORCH_VERSION}" in constraints


def test_base_image_tag_matches_the_pinned_torch_version():
    dockerfile = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert f"FROM pytorch/pytorch:{BASE_TORCH_VERSION}-cuda12.4" in dockerfile


def test_dockerfile_installs_requirements_under_the_constraints():
    dockerfile = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY docker/constraints.txt /tmp/constraints.txt" in dockerfile
    assert "pip install -c /tmp/constraints.txt -r /tmp/requirements.txt" in dockerfile


def test_requirements_do_not_list_torch_packages():
    lines = (DOCKER_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
    names = [line.strip().lower() for line in lines if line.strip() and not line.startswith("#")]
    assert not [n for n in names if n.startswith(("torch", "torchaudio", "torchvision"))]


def test_pip_check_ignores_only_the_known_base_image_ninja_note():
    output = "ninja 1.11.1.1 is not supported on this platform\n"
    assert blocking_pip_check_lines(output) == []


def test_pip_check_keeps_real_conflicts():
    output = (
        "ninja 1.11.1.1 is not supported on this platform\n"
        "pyannote-audio 3.4.0 has requirement torchaudio>=2.2.0, but you have torchaudio 1.0.\n"
    )
    assert blocking_pip_check_lines(output) == [
        "pyannote-audio 3.4.0 has requirement torchaudio>=2.2.0, but you have torchaudio 1.0."
    ]


def test_pip_check_keeps_unsupported_notes_for_other_packages():
    output = "numpy 1.26.4 is not supported on this platform\n"
    assert blocking_pip_check_lines(output) == ["numpy 1.26.4 is not supported on this platform"]


def test_pip_check_clean_output_has_no_blocking_lines():
    assert blocking_pip_check_lines("No broken requirements found.\n") == []
