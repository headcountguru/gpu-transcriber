"""Check a Hugging Face token and its access to the two gated pyannote models.

Mirrors cell 3b of the Kaggle notebook: ``GET /api/whoami-v2`` for the token and user name, then
``GET /api/models/<model>`` for each gated model, where a 200 means the user accepted the terms.
The token is sent only in the ``Authorization`` header; it is never put in a URL, a log line, an
exception message or the returned :class:`HfAccess`.
"""

import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
GATED_MODELS = (DIARIZATION_MODEL, SEGMENTATION_MODEL)
REQUEST_TIMEOUT_S = 15.0

# Statuses that mean "this token is not allowed to see this model" (terms not accepted).
_DENIED_STATUSES = (401, 403, 404)


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

    An invalid token returns ``valid=False`` without calling the model endpoints. Network
    failures and unexpected server replies raise :class:`HfCheckError`.
    """
    http = session if session is not None else requests.Session()
    headers = {"Authorization": f"Bearer {token}"}

    whoami = _get(http, f"{HF_API}/whoami-v2", headers)
    if whoami.status_code == 401:
        return HfAccess(valid=False, user=None, models=dict.fromkeys(GATED_MODELS, False))
    if whoami.status_code != 200:
        raise _unexpected(whoami.status_code)
    user = _user_name(whoami)

    models: dict[str, bool] = {}
    for model in GATED_MODELS:
        reply = _get(http, f"{HF_API}/models/{model}", headers)
        if reply.status_code == 200:
            models[model] = True
        elif reply.status_code in _DENIED_STATUSES:
            models[model] = False
        else:
            raise _unexpected(reply.status_code)
    return HfAccess(valid=True, user=user, models=models)


def _get(http: requests.Session, url: str, headers: dict[str, str]) -> requests.Response:
    try:
        return http.get(url, headers=headers, timeout=REQUEST_TIMEOUT_S)
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
