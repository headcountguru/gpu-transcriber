"""Check a Hugging Face token and its access to the two gated pyannote models.

``GET /api/whoami-v2`` gives the token's validity and user name. Model access is probed with
``HEAD https://huggingface.co/<model>/resolve/main/config.yaml`` (a gated file) using the token,
without following redirects: 200 or a 3xx redirect to the CDN means the user accepted the terms,
403 means they have not. Any other status (401 after a valid whoami, 404, 429, 5xx) is not a
verdict and raises :class:`HfCheckError`.

History: the Kaggle notebook's cell 3b used ``GET /api/models/<model>`` and treated 200 as
"access granted". That endpoint returns public metadata (200) for these models even to callers
without a token, so it can never say whether the terms were accepted; do not use it.

The token is sent only in the ``Authorization`` header; it is never put in a URL, a log line, an
exception message or the returned :class:`HfAccess`.
"""

import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

HF_SITE = "https://huggingface.co"
HF_API = f"{HF_SITE}/api"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
GATED_MODELS = (DIARIZATION_MODEL, SEGMENTATION_MODEL)
GATED_PROBE_FILE = "config.yaml"
REQUEST_TIMEOUT_S = 15.0

_TERMS_NOT_ACCEPTED_STATUS = 403


class HfCheckError(RuntimeError):
    """The Hugging Face check could not be completed. Never contains the token."""


@dataclass(frozen=True)
class HfAccess:
    """Outcome of a check. ``models`` maps each gated model id to whether access is granted."""

    valid: bool
    user: str | None
    models: dict[str, bool]


def check_hf_access(token: str, session: requests.Session | None = None) -> HfAccess:
    """Validate ``token`` and report access to each gated model.

    An invalid token returns ``valid=False`` without probing the models. Network failures and
    unexpected server replies raise :class:`HfCheckError`.
    """
    http = session if session is not None else requests.Session()
    headers = {"Authorization": f"Bearer {token}"}

    whoami = _send(http, "GET", f"{HF_API}/whoami-v2", headers)
    if whoami.status_code == 401:
        return HfAccess(valid=False, user=None, models=dict.fromkeys(GATED_MODELS, False))
    if whoami.status_code != 200:
        raise _unexpected(whoami.status_code)
    user = _user_name(whoami)

    models: dict[str, bool] = {}
    for model in GATED_MODELS:
        url = f"{HF_SITE}/{model}/resolve/main/{GATED_PROBE_FILE}"
        status = _send(http, "HEAD", url, headers).status_code
        if status == 200 or 300 <= status < 400:
            models[model] = True
        elif status == _TERMS_NOT_ACCEPTED_STATUS:
            models[model] = False
        else:
            raise _unexpected(status)
    return HfAccess(valid=True, user=user, models=models)


def _send(
    http: requests.Session, method: str, url: str, headers: dict[str, str]
) -> requests.Response:
    # Redirects are never followed: a 3xx to the CDN already answers the question, and following
    # it would download the file.
    try:
        return http.request(
            method, url, headers=headers, timeout=REQUEST_TIMEOUT_S, allow_redirects=False
        )
    except requests.RequestException as exc:
        # Log only the type: the message could echo request details.
        logger.warning("Hugging Face check failed (%s)", type(exc).__name__)
        raise HfCheckError(
            "Could not reach Hugging Face. Check your internet connection and try again."
        ) from None


def _user_name(reply: requests.Response) -> str | None:
    try:
        name = reply.json().get("name")
    except (ValueError, AttributeError):
        return None
    return name if isinstance(name, str) else None


def _unexpected(status: int) -> HfCheckError:
    logger.warning("Hugging Face check got unexpected status %s", status)
    return HfCheckError(
        f"Hugging Face answered with an unexpected status ({status}). Try again later."
    )
