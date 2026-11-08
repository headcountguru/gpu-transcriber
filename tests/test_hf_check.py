"""Tests for the Hugging Face token and gated-model access check (HTTP mocked)."""

import logging
from typing import Any

import pytest
import requests
import responses

from app.hf_check import (
    DIARIZATION_MODEL,
    SEGMENTATION_MODEL,
    HfAccess,
    HfCheckError,
    check_hf_access,
)

TOKEN = "hf_secret_token_value_456"
WHOAMI = "https://huggingface.co/api/whoami-v2"
DIARIZATION_FILE = f"https://huggingface.co/{DIARIZATION_MODEL}/resolve/main/config.yaml"
SEGMENTATION_FILE = f"https://huggingface.co/{SEGMENTATION_MODEL}/resolve/main/config.yaml"
CDN_URL = "https://cdn-lfs.example.test/some/blob"


def _whoami_ok() -> None:
    responses.get(WHOAMI, json={"name": "tom"}, status=200)


@responses.activate
def test_check_hf_access_all_granted_with_200_and_redirect() -> None:
    _whoami_ok()
    responses.head(DIARIZATION_FILE, status=200)
    responses.head(SEGMENTATION_FILE, status=302, headers={"Location": CDN_URL})

    access = check_hf_access(TOKEN)

    assert access == HfAccess(
        valid=True, user="tom", models={DIARIZATION_MODEL: True, SEGMENTATION_MODEL: True}
    )
    for call in responses.calls:
        assert call.request.headers["Authorization"] == f"Bearer {TOKEN}"
        assert call.request.url is not None and TOKEN not in call.request.url


@responses.activate
def test_check_hf_access_reports_denied_models() -> None:
    _whoami_ok()
    responses.head(DIARIZATION_FILE, status=302, headers={"Location": CDN_URL})
    responses.head(SEGMENTATION_FILE, status=403)

    access = check_hf_access(TOKEN)

    assert access.valid is True
    assert access.user == "tom"
    assert access.models == {DIARIZATION_MODEL: True, SEGMENTATION_MODEL: False}


@responses.activate
def test_probe_is_the_gated_file_not_public_metadata() -> None:
    """The /api/models/<id> endpoint answers 200 to everyone, so it must never be used."""
    _whoami_ok()
    responses.head(DIARIZATION_FILE, status=200)
    responses.head(SEGMENTATION_FILE, status=200)

    check_hf_access(TOKEN)

    probed = [(c.request.method, c.request.url) for c in responses.calls[1:]]
    assert probed == [("HEAD", DIARIZATION_FILE), ("HEAD", SEGMENTATION_FILE)]
    assert not any("/api/models/" in (c.request.url or "") for c in responses.calls)


@responses.activate
def test_redirects_are_not_followed() -> None:
    """A 302 to the CDN is an 'accepted' answer; following it would download the file."""
    _whoami_ok()
    responses.head(DIARIZATION_FILE, status=302, headers={"Location": CDN_URL})
    responses.head(SEGMENTATION_FILE, status=302, headers={"Location": CDN_URL})

    access = check_hf_access(TOKEN)

    assert all(access.models.values())
    assert not any(c.request.url == CDN_URL for c in responses.calls)


@responses.activate
def test_check_hf_access_invalid_token() -> None:
    responses.get(WHOAMI, json={"error": "Invalid credentials"}, status=401)

    access = check_hf_access(TOKEN)

    assert access == HfAccess(
        valid=False, user=None, models={DIARIZATION_MODEL: False, SEGMENTATION_MODEL: False}
    )
    # A bad token must not be sent to the model files.
    assert [call.request.url for call in responses.calls] == [WHOAMI]


@pytest.mark.parametrize("status", [401, 404, 429, 500, 503])
@responses.activate
def test_unexpected_model_status_raises_plain_error_not_denied(status: int) -> None:
    _whoami_ok()
    responses.head(DIARIZATION_FILE, status=status)

    with pytest.raises(HfCheckError) as excinfo:
        check_hf_access(TOKEN)

    assert str(status) in str(excinfo.value)
    assert TOKEN not in str(excinfo.value)


@responses.activate
def test_unexpected_whoami_status_raises_plain_error_without_token() -> None:
    responses.get(WHOAMI, status=500)

    with pytest.raises(HfCheckError) as excinfo:
        check_hf_access(TOKEN)

    assert "500" in str(excinfo.value)
    assert TOKEN not in str(excinfo.value)


@pytest.mark.parametrize("error", [requests.ConnectionError, requests.Timeout])
@responses.activate
def test_network_failure_raises_plain_error_without_token(
    error: type[requests.RequestException], caplog: pytest.LogCaptureFixture
) -> None:
    responses.get(WHOAMI, body=error(f"failed for token {TOKEN}"))

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(HfCheckError) as excinfo:
            check_hf_access(TOKEN)

    assert TOKEN not in str(excinfo.value)
    assert TOKEN not in repr(excinfo.value)
    assert excinfo.value.__suppress_context__
    assert TOKEN not in caplog.text


@responses.activate
def test_model_probe_network_failure_raises_plain_error() -> None:
    _whoami_ok()
    responses.head(DIARIZATION_FILE, body=requests.Timeout(f"slow for {TOKEN}"))

    with pytest.raises(HfCheckError) as excinfo:
        check_hf_access(TOKEN)

    assert TOKEN not in str(excinfo.value)


def test_hf_access_has_no_token_field_and_is_frozen() -> None:
    access = HfAccess(valid=True, user="tom", models={})
    assert TOKEN not in repr(access)
    assert "token" not in HfAccess.__dataclass_fields__
    with pytest.raises(AttributeError):
        access.valid = False  # type: ignore[misc]


@responses.activate
def test_injected_session_is_used_with_timeout_and_no_redirects() -> None:
    _whoami_ok()
    responses.head(DIARIZATION_FILE, status=200)
    responses.head(SEGMENTATION_FILE, status=200)

    seen: list[tuple[str, dict[str, Any]]] = []

    class SpySession(requests.Session):
        def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:  # type: ignore[override]
            seen.append((method, kwargs))
            return super().request(method, url, **kwargs)

    check_hf_access(TOKEN, session=SpySession())

    assert [m for m, _ in seen] == ["GET", "HEAD", "HEAD"]
    assert all(0 < kw["timeout"] <= 30 for _, kw in seen)
    assert all(kw["headers"]["Authorization"] == f"Bearer {TOKEN}" for _, kw in seen)
    assert all(kw["allow_redirects"] is False for m, kw in seen if m == "HEAD")
