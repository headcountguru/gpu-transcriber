"""Secret storage in Windows Credential Manager, through the ``keyring`` package.

Secrets (the Vast.ai API key and the Hugging Face token) are never written to settings files.
A secret value never appears in an exception message, a log line or a repr: backend errors are
replaced by a plain-words :class:`CredentialError` raised ``from None`` so the original
exception (whose text could echo the value) is not chained.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SERVICE = "GpuTranscriber"
VAST_API_KEY = "vast_api_key"
HF_TOKEN = "hf_token"


class CredentialError(RuntimeError):
    """The credential store could not be read or written. Never contains a secret value."""


class CredentialStore:
    """Get, set and delete named secrets.

    ``backend`` is any object with keyring's ``get_password``/``set_password``/
    ``delete_password`` methods. When omitted, the real ``keyring`` module is imported lazily
    on first use, so tests with a fake backend never touch Windows Credential Manager.
    """

    def __init__(self, service: str = DEFAULT_SERVICE, backend: object | None = None) -> None:
        self._service = service
        self._backend = backend

    def __repr__(self) -> str:
        return f"CredentialStore(service={self._service!r})"

    def get(self, name: str) -> str | None:
        """Return the stored secret, or ``None`` if it was never saved."""
        backend = self._resolve_backend()
        try:
            value = backend.get_password(self._service, name)
        except Exception as exc:
            raise self._failure("read", name, exc) from None
        return value if isinstance(value, str) else None

    def set(self, name: str, value: str) -> None:
        """Save (or replace) a secret."""
        backend = self._resolve_backend()
        try:
            backend.set_password(self._service, name, value)
        except Exception as exc:
            raise self._failure("save", name, exc) from None

    def set_all(self, values: dict[str, str]) -> None:
        """Save several secrets: all of them, or none (earlier ones are put back on failure).

        Raises ``CredentialError`` for the save that failed. If putting an earlier secret back
        also fails, that is logged (by name only) and the original failure is still raised.
        """
        previous = {name: self.get(name) for name in values}
        written: list[str] = []
        try:
            for name, value in values.items():
                self.set(name, value)
                written.append(name)
        except CredentialError:
            for name in reversed(written):
                old = previous[name]
                try:
                    if old is None:
                        self.delete(name)
                    else:
                        self.set(name, old)
                except CredentialError:
                    logger.error("Could not put back the previous %s after a failed save", name)
            raise

    def delete(self, name: str) -> None:
        """Remove a secret. Removing one that does not exist is not an error."""
        backend = self._resolve_backend()
        try:
            if backend.get_password(self._service, name) is None:
                return
            backend.delete_password(self._service, name)
        except Exception as exc:
            raise self._failure("remove", name, exc) from None

    def _resolve_backend(self) -> Any:
        if self._backend is None:
            try:
                import keyring
            except ImportError:
                raise CredentialError(
                    "Secure storage is unavailable: the 'keyring' package is not installed."
                ) from None
            self._backend = keyring
        return self._backend

    @staticmethod
    def _failure(action: str, name: str, exc: Exception) -> CredentialError:
        # Log only the exception type: its message could echo the secret value.
        logger.warning("Credential store could not %s %r (%s)", action, name, type(exc).__name__)
        return CredentialError(
            f"Could not {action} the saved {name!r} in Windows Credential Manager. "
            "Check that secure storage is available on this PC."
        )
