"""Tests for the keyring-backed credential store, using an in-memory fake backend only."""

import logging

import pytest

from app.credentials import HF_TOKEN, VAST_API_KEY, CredentialError, CredentialStore

SECRET = "hf_super_secret_value_123"


class FakeBackend:
    """Minimal stand-in for the ``keyring`` module; never touches Windows Credential Manager."""

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, name: str) -> str | None:
        return self.data.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self.data[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        if (service, name) not in self.data:
            raise KeyError("not found")  # keyring raises PasswordDeleteError for a missing name
        del self.data[(service, name)]


class BrokenBackend:
    """Backend whose every call fails with a message that embeds the secret."""

    def get_password(self, service: str, name: str) -> str | None:
        raise RuntimeError(f"no backend, value was {SECRET}")

    def set_password(self, service: str, name: str, value: str) -> None:
        raise RuntimeError(f"no backend, value was {value}")

    def delete_password(self, service: str, name: str) -> None:
        raise RuntimeError("no backend")


def test_names_are_the_documented_ones() -> None:
    assert VAST_API_KEY == "vast_api_key"
    assert HF_TOKEN == "hf_token"


def test_set_get_delete_roundtrip() -> None:
    backend = FakeBackend()
    store = CredentialStore(backend=backend)

    store.set(HF_TOKEN, SECRET)
    assert store.get(HF_TOKEN) == SECRET
    assert backend.data == {("GpuTranscriber", "hf_token"): SECRET}

    store.delete(HF_TOKEN)
    assert store.get(HF_TOKEN) is None


def test_get_missing_returns_none() -> None:
    assert CredentialStore(backend=FakeBackend()).get(VAST_API_KEY) is None


def test_delete_missing_does_not_raise() -> None:
    CredentialStore(backend=FakeBackend()).delete(VAST_API_KEY)


def test_custom_service_name_is_used() -> None:
    backend = FakeBackend()
    CredentialStore(service="Other", backend=backend).set(VAST_API_KEY, "k")
    assert ("Other", "vast_api_key") in backend.data


def test_backend_failure_raises_plain_error_without_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = CredentialStore(backend=BrokenBackend())
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(CredentialError) as excinfo:
            store.set(HF_TOKEN, SECRET)
        with pytest.raises(CredentialError):
            store.get(HF_TOKEN)
        with pytest.raises(CredentialError):
            store.delete(HF_TOKEN)

    assert SECRET not in str(excinfo.value)
    assert SECRET not in repr(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__
    assert SECRET not in caplog.text


def test_repr_does_not_contain_secret() -> None:
    store = CredentialStore(backend=FakeBackend())
    store.set(HF_TOKEN, SECRET)
    assert SECRET not in repr(store)


def test_default_backend_is_the_keyring_module_loaded_lazily() -> None:
    import keyring

    store = CredentialStore()  # must not touch the real credential manager on construction
    assert store._resolve_backend() is keyring


class FailAfterFirstWrite(FakeBackend):
    """The first save works; every later save or removal fails."""

    def __init__(self) -> None:
        super().__init__()
        self.writes = 0

    def set_password(self, service: str, name: str, value: str) -> None:
        self.writes += 1
        if self.writes > 1:
            raise RuntimeError(f"locked while saving {value}")
        super().set_password(service, name, value)

    def delete_password(self, service: str, name: str) -> None:
        raise RuntimeError("locked")


def test_set_all_saves_every_value() -> None:
    backend = FakeBackend()
    CredentialStore(backend=backend).set_all({VAST_API_KEY: "v-key", HF_TOKEN: "h-token"})

    assert sorted(backend.data.values()) == ["h-token", "v-key"]


def test_set_all_reports_a_failed_put_back_by_name_only(caplog: pytest.LogCaptureFixture) -> None:
    backend = FailAfterFirstWrite()
    store = CredentialStore(backend=backend)

    with caplog.at_level(logging.ERROR), pytest.raises(CredentialError) as err:
        store.set_all({VAST_API_KEY: "v-key-SECRET", HF_TOKEN: "h-token-SECRET"})

    assert "SECRET" not in str(err.value)
    assert "SECRET" not in caplog.text
    assert VAST_API_KEY in caplog.text  # the first key could not be removed again
