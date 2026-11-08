"""Tests for the Hugging Face token and gated-model access check (HTTP mocked)."""

import logging

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
DIARIZATION_URL = f"https://huggingface.co/api/models/{DIARIZATION_MODEL}"
SEGMENTATION_URL = f"https://huggingface.co/api/models/{SEGMENTATION_MODEL}"


@responses.activate
def test_check_hf_access_all_granted() -> None:
    responses.get(WHOAMI, json={"name": "tom"}, status=200)
    responses.get(DIARIZATION_URL, json={}, status=200)
    responses.get(SEGMENTATION_URL, json={}, status=200)

    access = check_hf_access(TOKEN)

    assert access == HfAccess(
        valid=True, user="tom", models={DIARIZATION_MODEL: True, SEGMENTATION_MODEL: True}
    )
    for call in responses.calls:
        assert call.request.headers["Authorization"] == f"Bearer {TOKEN}"
        assert call.request.url is not None and TOKEN not in call.request.url


@responses.activate
def test_check_hf_access_reports_denied_models() -> None:
    responses.get(WHOAMI, json={"name": "tom"}, status=200)
    responses.get(DIARIZATION_URL, json={}, status=200)
    responses.get(SEGMENTATION_URL, json={"error": "gated"}, status=403)

    access = check_hf_access(TOKEN)

    assert access.valid is True
    assert access.user == "tom"
    assert access.models == {DIARIZATION_MODEL: True, SEGMENTATION_MODEL: False}


@responses.activate
def test_check_hf_access_unauthorized_model_is_denied() -> None:
    responses.get(WHOAMI, json={"name": "tom"}, status=200)
    responses.get(DIARIZATION_URL, json={}, status=401)
    responses.get(SEGMENTATION_URL, json={}, status=404)

    access = check_hf_access(TOKEN)

    assert access.models == {DIARIZATION_MODEL: False, SEGMENTATION_MODEL: False}


@responses.activate
def test_check_hf_access_invalid_token() -> None:
    responses.get(WHOAMI, json={"error": "Invalid credentials"}, status=401)

    access = check_hf_access(TOKEN)

    assert access == HfAccess(
        valid=False, user=None, models={DIARIZATION_MODEL: False, SEGMENTATION_MODEL: False}
    )
    # A bad token must not be sent to the model endpoints.
    assert [call.request.url for call in responses.calls] == [WHOAMI]


@responses.activate
def test_unexpected_status_raises_plain_error_without_token() -> None:
    responses.get(WHOAMI, status=500)

    with pytest.raises(HfCheckError) as excinfo:
        check_hf_access(TOKEN)

    assert "500" in str(excinfo.value)
    assert TOKEN not in str(excinfo.value)


@responses.activate
def test_unexpected_model_status_raises_plain_error() -> None:
    responses.get(WHOAMI, json={"name": "tom"}, status=200)
    responses.get(DIARIZATION_URL, status=503)

    with pytest.raises(HfCheckError) as excinfo:
        check_hf_access(TOKEN)

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


def test_hf_access_has_no_token_field_and_is_frozen() -> None:
    access = HfAccess(valid=True, user="tom", models={})
    assert TOKEN not in repr(access)
    assert "token" not in HfAccess.__dataclass_fields__
    with pytest.raises(AttributeError):
        access.valid = False  # type: ignore[misc]


@responses.activate
def test_injected_session_is_used_with_timeout() -> None:
    responses.get(WHOAMI, json={"name": "tom"}, status=200)
    responses.get(DIARIZATION_URL, json={}, status=200)
    responses.get(SEGMENTATION_URL, json={}, status=200)

    seen: list[object] = []

    class SpySession(requests.Session):
        def get(self, url, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(kwargs.get("timeout"))
            return super().get(url, **kwargs)

    check_hf_access(TOKEN, session=SpySession())

    assert len(seen) == 3
    assert all(isinstance(t, (int, float)) and 0 < t <= 30 for t in seen)
