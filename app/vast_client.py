"""Vast.ai REST client and offer ranking.

This is the app's only Vast.ai client. The API key is a secret: it is sent only in the
``Authorization`` header, never logged, and scrubbed from every error message.

Documentation relied on (fetched 2026-11-06, Vast.ai REST docs):
  - https://docs.vast.ai/api-reference/introduction
  - https://docs.vast.ai/api-reference/authentication            (Bearer token)
  - https://docs.vast.ai/api-reference/rate-limits-and-errors    (error body shapes, HTTP 429)
  - https://docs.vast.ai/api-reference/search/search-offers      (POST /api/v0/bundles/)
  - https://docs.vast.ai/api-reference/instances/create-instance (PUT /api/v0/asks/{id}/)
  - https://docs.vast.ai/api-reference/creating-instances-with-api (env must be a JSON object)
  - https://docs.vast.ai/api-reference/instances/show-instance   (GET /api/v0/instances/{id}/)
  - https://docs.vast.ai/api-reference/instances/show-instances  (GET /api/v1/instances, paged)
  - https://docs.vast.ai/api-reference/instances/destroy-instance (DELETE /api/v0/instances/{id}/)
  - https://docs.vast.ai/api-reference/accounts/show-user        (GET /api/v0/users/current/)
  - https://docs.vast.ai/api-reference/accounts/create-ssh-key   (POST /api/v0/ssh/)
  - https://docs.vast.ai/cli/reference/search-offers             (units of ``inet_down``)

Unit conversions: the REST API reports ``gpu_ram`` in MB per GPU (docs example: ``gte 24000``
for a 24 GB card) so ``gpu_ram_gb = gpu_ram / 1000`` (a "16 GB" card reports ~16376 MB and must
still pass a 16 GB minimum). ``inet_down`` is already Mb/s (CLI docs), so it is used unchanged.

# UNCONFIRMED: datacenter flag. Offers carry ``hosting_type`` (docs example shows 0); we treat
# UNCONFIRMED: ``hosting_type >= 1`` as datacenter. Task 9 should check a real datacenter offer.
# UNCONFIRMED: ``gpu_ram`` divisor (1000 vs 1024) and the ``inet_down`` unit (Mb/s vs MB/s: the
# UNCONFIRMED: REST page says MB/s, the CLI page says Mb/s; we follow the CLI).
# CONFIRMED live (2026-11-07, credit-only account): ``GET /users/current/`` returns ``credit: 10``
# CONFIRMED live: and ``balance: 0``; ``get_balance`` therefore prefers ``credit`` over ``balance``.
# UNCONFIRMED: ``POST /bundles/`` ``limit`` ceiling and whether ``order`` is honoured as sent.
# UNCONFIRMED: trailing slashes on ``/bundles/``, ``/asks/{id}/``, ``/instances/{id}/``,
# UNCONFIRMED: ``/users/current/`` follow the doc examples; ``/ssh/`` is from the CLI convention
# UNCONFIRMED: (the docs list ``/api/v0/ssh``).
# UNCONFIRMED: ``show instances`` is documented only as ``/api/v1/instances`` (paged); we derive
# UNCONFIRMED: the v1 URL from ``base_url``. Legacy ``GET /api/v0/instances/`` is the fallback.
# UNCONFIRMED: what ``GET /instances/{id}/`` returns for a destroyed instance (404 vs 200 with an
# UNCONFIRMED: empty ``instances``); both are treated as "gone". ``instances`` is a single object.
# UNCONFIRMED: ``actual_status`` is null while an instance is still being set up; we report that
# UNCONFIRMED: as ``"pending"``. Other values (``loading``, ``running``, ``exited``, ``offline``,
# UNCONFIRMED: ``unknown``) are passed through; the docs say ``exited``/``unknown``/``offline``
# UNCONFIRMED: never become ``running``.
# UNCONFIRMED: registering an SSH key that already exists may return an error; not handled.
# UNCONFIRMED: ``runtype`` is left at the default (``ssh``, proxied); ``ssh_direct`` was not used.
# UNCONFIRMED: ``onstart`` is capped at 4048 characters (docs); longer scripts are rejected here.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

from app.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://console.vast.ai/api/v0"
REQUEST_TIMEOUT_S = 30.0
SEARCH_LIMIT = 500
LIST_PAGE_SIZE = 25
MAX_LIST_PAGES = 100
MAX_ONSTART_CHARS = 4048
ERROR_EXCERPT_CHARS = 200
MB_PER_GB = 1000.0
PENDING_STATUS = "pending"


class VastError(RuntimeError):
    """A Vast.ai request failed. The message is plain words, safe to show to the user."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class VastAmbiguousCreateError(VastError):
    """Renting a machine timed out after the request was sent, so one may exist.

    Callers must look for instances with their label (``list_instances``) before retrying,
    otherwise a retry could rent a second machine and leave the first one billing.
    """

    maybe_created: bool = True

    def __init__(self) -> None:
        super().__init__("Vast.ai did not answer in time; a machine may have been rented")


@dataclass(frozen=True)
class Offer:
    """A rentable machine, with Vast.ai's units converted (GB, Mb/s, dollars per hour)."""

    id: int
    gpu_name: str
    gpu_ram_gb: float
    dph_total: float
    inet_down_mbps: float
    reliability: float
    datacenter: bool
    cuda_max_good: float


@dataclass(frozen=True)
class Instance:
    """A rented instance as reported by Vast.ai."""

    id: int
    status: str
    ssh_host: str | None
    ssh_port: int | None
    dph_total: float
    label: str | None


def rank_offers(offers: list[Offer], settings: Settings) -> list[Offer]:
    """Drop offers that fail a minimum or the price ceiling, then order the rest.

    Datacenter machines come first when ``settings.prefer_datacenter``; within each group
    the cheapest comes first. An offer with a missing or unusable value is dropped.
    """
    fitting = [offer for offer in offers if _fits(offer, settings)]

    def sort_key(offer: Offer) -> tuple[int, float, int]:
        group = 0 if (settings.prefer_datacenter and offer.datacenter) else 1
        return (group, offer.dph_total, offer.id)

    return sorted(fitting, key=sort_key)


def _fits(offer: Offer, settings: Settings) -> bool:
    try:
        failures = [
            (offer.gpu_ram_gb < settings.min_vram_gb, "not enough GPU memory"),
            (offer.reliability < settings.min_reliability, "reliability too low"),
            (offer.inet_down_mbps < settings.min_down_mbps, "download too slow"),
            (offer.cuda_max_good < settings.min_cuda, "CUDA too old"),
            (offer.dph_total > settings.max_price_per_hour, "over the price ceiling"),
        ]
    except TypeError:
        logger.debug("Dropping offer %s: a needed value is missing", offer.id)
        return False
    for failed, reason in failures:
        if failed:
            logger.debug("Dropping offer %s: %s", offer.id, reason)
            return False
    return True


def _parse_offer(raw: Any) -> Offer | None:
    """Build an Offer from a search result, or None if a needed field is missing."""
    try:
        return Offer(
            id=int(raw["id"]),
            gpu_name=str(raw["gpu_name"]),
            gpu_ram_gb=float(raw["gpu_ram"]) / MB_PER_GB,
            dph_total=float(raw["dph_total"]),
            inet_down_mbps=float(raw["inet_down"]),
            reliability=float(raw["reliability"]),
            datacenter=int(raw.get("hosting_type") or 0) >= 1,
            cuda_max_good=float(raw["cuda_max_good"]),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        offer_id = raw.get("id") if isinstance(raw, dict) else None
        logger.debug("Skipping offer %s: missing or unreadable field", offer_id)
        return None


def _parse_instance(raw: Any) -> Instance:
    try:
        port = raw.get("ssh_port")
        return Instance(
            id=int(raw["id"]),
            status=str(raw.get("actual_status") or PENDING_STATUS),
            ssh_host=str(raw["ssh_host"]) if raw.get("ssh_host") else None,
            ssh_port=int(port) if port else None,
            dph_total=float(raw.get("dph_total") or 0.0),
            label=str(raw["label"]) if raw.get("label") else None,
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise VastError(
            "Vast.ai sent back an instance description that could not be read."
        ) from None


class VastClient:
    """Thin client for the Vast.ai REST API. Every failure raises ``VastError``."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._session = session if session is not None else requests.Session()
        self._base_url = base_url.rstrip("/")
        self._v1_url = re.sub(r"/v0$", "/v1", self._base_url)

    def __repr__(self) -> str:
        return f"VastClient(base_url={self._base_url!r})"

    # --- public API -------------------------------------------------------------------------

    def search_offers(self, settings: Settings | None = None) -> list[Offer]:
        """Return rentable, verified, single-GPU on-demand offers, cheapest first.

        When ``settings`` is given, its minimums and price ceiling are also sent to Vast.ai as
        search filters, so cheap unsuitable machines cannot crowd suitable ones out of the
        result. ``rank_offers`` remains the final client-side check.
        """
        body: dict[str, Any] = {
            "type": "ondemand",
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"eq": 1},
            "limit": SEARCH_LIMIT,
            "order": [["dph_total", "asc"]],
        }
        if settings is not None:
            body.update(
                {
                    "gpu_ram": {"gte": settings.min_vram_gb * MB_PER_GB},
                    "dph_total": {"lte": settings.max_price_per_hour},
                    "reliability": {"gte": settings.min_reliability},
                    "inet_down": {"gte": settings.min_down_mbps},
                    "cuda_max_good": {"gte": settings.min_cuda},
                }
            )
        data = self._call("POST", f"{self._base_url}/bundles/", "search for machines", json=body)
        raw_offers = data.get("offers") if isinstance(data, dict) else None
        parsed = [_parse_offer(raw) for raw in raw_offers or []]
        return [offer for offer in parsed if offer is not None]

    def create_instance(
        self,
        offer_id: int,
        image: str,
        label: str,
        disk_gb: int,
        onstart: str,
        env: dict[str, str],
    ) -> int:
        """Rent the machine behind ``offer_id`` and return the new instance id."""
        what = f"rent machine offer {offer_id}"
        if len(onstart) > MAX_ONSTART_CHARS:
            raise VastError(f"Could not {what}: the start-up script is too long for Vast.ai.")
        body = {
            "image": image,
            "label": label,
            "disk": disk_gb,
            "onstart": onstart,
            "env": env,
            "cancel_unavail": True,
        }
        data = self._call(
            "PUT", f"{self._base_url}/asks/{offer_id}/", what, json=body, ambiguous_timeout=True
        )
        new_id = data.get("new_contract") if isinstance(data, dict) else None
        if not isinstance(new_id, int) or isinstance(new_id, bool):
            raise VastError(f"Could not {what}: Vast.ai did not return an instance id.")
        logger.info("Rented offer %s as instance %s", offer_id, new_id)
        return new_id

    def get_instance(self, instance_id: int) -> Instance | None:
        """Return the instance, or None if it no longer exists."""
        what = f"check instance {instance_id}"
        data = self._call(
            "GET", f"{self._base_url}/instances/{instance_id}/", what, allow_not_found=True
        )
        raw = data.get("instances") if isinstance(data, dict) else None
        if isinstance(raw, list):
            raw = next((r for r in raw if isinstance(r, dict) and r.get("id") == instance_id), None)
        if not raw:
            return None
        return _parse_instance(raw)

    def list_instances(self, label_prefix: str) -> list[Instance]:
        """Return every instance whose label starts with ``label_prefix``."""
        what = "list your instances"
        found: list[Instance] = []
        token: str | None = None
        for _ in range(MAX_LIST_PAGES):
            params: dict[str, Any] = {"limit": LIST_PAGE_SIZE}
            if token:
                params["after_token"] = token
            data = self._call("GET", f"{self._v1_url}/instances", what, params=params)
            page = data.get("instances") if isinstance(data, dict) else None
            for raw in page or []:
                label = raw.get("label") if isinstance(raw, dict) else None
                if isinstance(label, str) and label.startswith(label_prefix):
                    found.append(_parse_instance(raw))
            token = data.get("next_token") if isinstance(data, dict) else None
            if not token:
                return found
        raise VastError("Could not list your instances: Vast.ai returned too many pages.")

    def destroy_instance(self, instance_id: int) -> None:
        """Permanently destroy the instance (stops billing)."""
        what = f"destroy instance {instance_id}"
        self._call("DELETE", f"{self._base_url}/instances/{instance_id}/", what, require_json=False)
        logger.info("Destroyed instance %s", instance_id)

    def get_balance(self) -> float:
        """Return the account's remaining credit in dollars."""
        data = self._call("GET", f"{self._base_url}/users/current/", "read your Vast.ai balance")
        value = None
        if isinstance(data, dict):
            # A live credit-only account reports prepaid money as "credit" and a "balance" of 0,
            # so "credit" wins whenever present; "balance" (the documented field) is the fallback.
            value = data.get("credit") if data.get("credit") is not None else data.get("balance")
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise VastError(
                "Could not read your Vast.ai balance: the answer had no balance in it."
            ) from None

    def register_ssh_key(self, public_key: str) -> None:
        """Add an SSH public key to the account so new instances accept it."""
        self._call(
            "POST",
            f"{self._base_url}/ssh/",
            "register the SSH key",
            json={"ssh_key": public_key},
            require_json=False,
        )

    # --- plumbing ---------------------------------------------------------------------------

    def _call(
        self,
        method: str,
        url: str,
        what: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
        allow_not_found: bool = False,
        require_json: bool = True,
        ambiguous_timeout: bool = False,
    ) -> Any:
        """Send a request and return the decoded JSON body (None for an allowed 404).

        With ``ambiguous_timeout`` (used for renting), a read timeout means the request was sent
        but its outcome is unknown, so it raises ``VastAmbiguousCreateError``.
        """
        try:
            response = self._session.request(
                method,
                url,
                json=json,
                params=params,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=REQUEST_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            # Deliberately not chained: the original message could echo request details.
            logger.warning("Vast.ai request to %s failed: %s", what, type(exc).__name__)
            if ambiguous_timeout and isinstance(exc, requests.ReadTimeout):
                raise VastAmbiguousCreateError() from None
            raise VastError(
                f"Could not {what}: could not reach Vast.ai. Check the internet connection."
            ) from None

        if allow_not_found and response.status_code == 404:
            return None
        if not response.ok:
            raise self._http_error(what, response)

        try:
            data = response.json()
        except ValueError:
            if require_json:
                raise VastError(
                    f"Could not {what}: Vast.ai sent an answer that is not readable."
                ) from None
            return None
        if isinstance(data, dict) and data.get("success") is False:
            raise VastError(f"Could not {what}: Vast.ai refused. {self._excerpt(response)}".strip())
        return data

    def _http_error(self, what: str, response: requests.Response) -> VastError:
        status = response.status_code
        if status in (401, 403):
            message = f"Could not {what}: Vast.ai did not accept the API key. Check it in Settings."
        elif status == 429:
            message = (
                f"Could not {what}: too many requests were sent to Vast.ai. Try again shortly."
            )
        else:
            message = (
                f"Could not {what}: Vast.ai reported an error (HTTP {status}). "
                f"{self._excerpt(response)}"
            ).strip()
        return VastError(message, status_code=status)

    def _excerpt(self, response: requests.Response) -> str:
        """A short, key-free piece of the response, for error messages."""
        text = ""
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            for key in ("msg", "message", "error", "detail"):
                if isinstance(body.get(key), str) and body[key].strip():
                    text = body[key]
                    break
        if not text:
            text = response.text or ""
        text = " ".join(text.split())
        if self._api_key:
            text = text.replace(self._api_key, "***")
        if len(text) > ERROR_EXCERPT_CHARS:
            text = text[:ERROR_EXCERPT_CHARS] + "..."
        return text
